# Backend — Enterprise AI Assistant

FastAPI + LangGraph backend. The gradeable core: `POST /ask` runs every question through
the orchestration spine (router → agent → tools → generate) with SQLite-backed
conversation memory.

## Setup

```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env        # then add your OPENAI_API_KEY
```

## Run

```bash
uvicorn app.main:app --reload
```

- `GET  /health`            — liveness (works without an API key)
- `POST /auth/register`     — `{ "email": "...", "password": "..." }` → creates a user
- `POST /auth/login`        — returns `{ "access_token": "...", "token_type": "bearer" }`
- `POST /auth/logout`       — 🔒 stateless (client drops the token)
- `POST /ask`               — 🔒 `{ "question": "...", "session_id": "optional" }`
- `POST /documents/upload`  — 🔒 multipart file upload (`.pdf`, `.txt`, `.md`; ≤ 10 MB) → chunk → embed → index
- `GET  /documents`         — 🔒 list the current user's documents
- 🔒 = requires `Authorization: Bearer <token>`
- Interactive docs: http://127.0.0.1:8000/docs

Example:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Create a ticket: the VPN keeps disconnecting for finance."}'
```

The response surfaces `intent`, `tool_used`, and `sources` so the routed path is visible.

## Test

```bash
cd backend
pytest
```

## Scope so far

- **Phase 1** — backend core: `/ask` through the LangGraph spine (router → agent → tools →
  generate) with SQLite-backed memory. Tools: `create_ticket` (persisted), `fetch_employee_info`,
  `fetch_customer_info`, `generate_report`.
- **Phase 2** — RAG: document upload → chunk (~1k/150) → embed (`text-embedding-3-small`) → Chroma,
  with a Retrieve node grounding `knowledge` questions (top-k=4, filtered by `user_id`). Answers carry
  `sources`.
- **Phase 3** — JWT auth (register/login/logout) + multi-tenancy: `/ask` and `/documents*` are
  protected; the authenticated user (`str(user.id)`) scopes both conversation memory and RAG
  retrieval, so users never see each other's documents or history.
- **Phase 4** — guardrails + resilience: a first-node guardrail (regex pre-filter + optional LLM
  classifier) blocks prompt injection / abuse with a templated reply (`intent: "blocked"`);
  retry/backoff (tenacity) on transient provider errors; honest empty-retrieval degradation;
  tool failures and provider outages never 500. Toggle the LLM layer with `ENABLE_GUARDRAIL_LLM`.

Example: `register` → `login` → use the returned token as `Authorization: Bearer <token>` on `/ask`
and `/documents`.
