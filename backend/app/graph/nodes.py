"""Graph nodes: guardrail, router, retrieve, agent, tools, generate.

Flow: guardrail -> router -> [retrieve ->] agent -> [tools -> agent]* -> generate.
The guardrail short-circuits straight to END on blocked input.
"""
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.config import get_settings
from app.graph.guardrail import BLOCK_MESSAGES, GUARDRAIL_LLM_SYSTEM, regex_screen
from app.graph.llm import get_chat_llm, invoke_with_retry
from app.graph.state import GraphState
from app.rag.retrieve import retrieve_context
from app.tools.business import ALL_TOOLS, TOOLS_BY_NAME

logger = logging.getLogger(__name__)

VALID_INTENTS = {"action", "knowledge", "general"}

ROUTER_SYSTEM = (
    "You are an intent router for an enterprise assistant. Classify the user's latest "
    "message into exactly one label:\n"
    "- action: the user wants to perform an operation — create/file a support ticket, "
    "look up an employee or customer record, or generate a report.\n"
    "- knowledge: a factual question that should be answered from the user's uploaded "
    "documents or general knowledge.\n"
    "- general: greetings, small talk, or anything that is neither of the above.\n"
    "Respond with ONLY the single label word: action, knowledge, or general."
)

AGENT_SYSTEM = (
    "You are a helpful enterprise assistant. When a tool is available and relevant, call "
    "it instead of guessing; never fabricate employee/customer data or ticket ids. If a "
    "request is ambiguous or missing required information, ask a brief clarifying question "
    "rather than acting. Be concise."
)

KNOWLEDGE_CONTEXT_TEMPLATE = (
    "Use the following excerpts from the user's uploaded documents to answer the "
    "question. If the answer is not contained in them, say you don't have that "
    "information in the provided documents — do not invent it.\n\n{context}"
)

KNOWLEDGE_NO_CONTEXT = (
    "No documents were retrieved for this question. If it needs information specific to "
    "the user's company or uploaded documents, tell them you don't have that in their "
    "documents rather than guessing. If it is a general question, you may answer briefly."
)


def _last_human(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def guardrail_node(state: GraphState) -> dict:
    """First node: block empty/oversized/injection/out-of-scope input with a templated,
    honest response — no tool call, no fabrication (D9)."""
    text = _last_human(state["messages"])
    reason = regex_screen(text)  # fast, deterministic, no LLM

    if reason is None and get_settings().enable_guardrail_llm:
        try:
            verdict = invoke_with_retry(
                get_chat_llm(),
                [SystemMessage(content=GUARDRAIL_LLM_SYSTEM), HumanMessage(content=text)],
            )
            if (verdict.content or "").strip().lower().startswith("block"):
                reason = "policy"
        except Exception:  # fail-open: a flaky check never blocks legitimate use
            logger.exception("Guardrail LLM check failed; allowing")

    if reason:
        message = BLOCK_MESSAGES.get(reason, BLOCK_MESSAGES["policy"])
        return {
            "blocked": True,
            "intent": "blocked",
            "answer": message,
            "messages": [AIMessage(content=message)],
            "tool_used": None,
            "sources": [],
        }
    return {"blocked": False}


def route_after_guardrail(state: GraphState) -> str:
    return "blocked" if state.get("blocked") else "router"


def router_node(state: GraphState) -> dict:
    llm = get_chat_llm()
    messages = [SystemMessage(content=ROUTER_SYSTEM), *state["messages"]]
    result = invoke_with_retry(llm, messages)
    label = (result.content or "").strip().lower()
    intent = next((i for i in VALID_INTENTS if i in label), "general")
    return {"intent": intent}


def route_after_router(state: GraphState) -> str:
    # D8/D5: knowledge questions retrieve first; action/general skip RAG.
    return "retrieve" if state.get("intent") == "knowledge" else "agent"


def retrieve_node(state: GraphState) -> dict:
    # retrieve_context degrades to ([], []) on store errors; embedding/key errors still
    # propagate (handled at the API boundary) so a missing key surfaces as a clean 503.
    chunks, sources = retrieve_context(state["user_id"], _last_human(state["messages"]))
    return {"context": chunks, "sources": sources}


def agent_node(state: GraphState) -> dict:
    llm = get_chat_llm()
    messages = [SystemMessage(content=AGENT_SYSTEM)]
    if state.get("intent") == "knowledge":
        context = state.get("context")
        if context:
            messages.append(
                SystemMessage(content=KNOWLEDGE_CONTEXT_TEMPLATE.format(context="\n\n---\n\n".join(context)))
            )
        else:
            messages.append(SystemMessage(content=KNOWLEDGE_NO_CONTEXT))  # honest degradation
    messages.extend(state["messages"])
    # D8: bind tools only on the action path; knowledge/general get no tools.
    if state.get("intent") == "action":
        llm = llm.bind_tools(ALL_TOOLS)
    ai_message = invoke_with_retry(llm, messages)
    return {"messages": [ai_message]}


def route_after_agent(state: GraphState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "generate"


def tool_node(state: GraphState) -> dict:
    last = state["messages"][-1]
    outputs: list[ToolMessage] = []
    tool_used = state.get("tool_used")
    for call in last.tool_calls:
        tool = TOOLS_BY_NAME.get(call["name"])
        if tool is None:
            content = f"Error: unknown tool '{call['name']}'."
        else:
            try:
                content = tool.invoke(call["args"])
                tool_used = call["name"]
            except Exception as exc:  # a failing tool must not crash the graph
                logger.exception("Tool '%s' failed", call["name"])
                content = (
                    f"Error: the '{call['name']}' action could not be completed ({exc}). "
                    "Inform the user briefly and do not retry automatically."
                )
        outputs.append(ToolMessage(content=str(content), tool_call_id=call["id"]))
    return {"messages": outputs, "tool_used": tool_used}


def generate_node(state: GraphState) -> dict:
    """Finalize the answer. The agent has already composed prose (after seeing any tool
    results on the loop-back), so we extract it here. This is also where retrieved
    `sources` will be attached in Phase 2."""
    last = state["messages"][-1]
    answer = last.content if isinstance(last, (AIMessage, HumanMessage)) else str(last.content)
    return {"answer": answer or ""}
