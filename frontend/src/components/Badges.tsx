// Surfaces the workflow on every assistant turn: routed intent, the business tool that
// fired (if any), and retrieved sources. This visibility is intentional (spec) — it makes
// the LangGraph spine legible in the demo.
import type { AskResponse } from "@/lib/api";

const INTENT_STYLES: Record<string, string> = {
  action: "bg-violet-100 text-violet-800 ring-violet-200",
  knowledge: "bg-sky-100 text-sky-800 ring-sky-200",
  general: "bg-slate-100 text-slate-700 ring-slate-200",
  blocked: "bg-rose-100 text-rose-800 ring-rose-200",
};

function Pill({ className, children }: { className: string; children: React.ReactNode }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${className}`}
    >
      {children}
    </span>
  );
}

function sourceLabel(source: Record<string, unknown>, i: number): string {
  const name =
    (source.filename as string) ||
    (source.source as string) ||
    (source.document_id != null ? `doc ${source.document_id}` : "");
  const chunk = source.chunk_index != null ? ` #${source.chunk_index}` : "";
  return name ? `${name}${chunk}` : `source ${i + 1}`;
}

export function MetaBadges({ result }: { result: AskResponse }) {
  return (
    <div className="mt-2 flex flex-wrap items-center gap-1.5">
      <Pill className={INTENT_STYLES[result.intent] ?? INTENT_STYLES.general}>
        intent: {result.intent}
      </Pill>
      {result.tool_used && (
        <Pill className="bg-emerald-100 text-emerald-800 ring-emerald-200">
          tool: {result.tool_used}
        </Pill>
      )}
      {result.sources?.length > 0 && (
        <Pill className="bg-amber-100 text-amber-800 ring-amber-200">
          {result.sources.length} source{result.sources.length > 1 ? "s" : ""}
        </Pill>
      )}
      {result.sources?.length > 0 && (
        <span className="w-full text-xs text-slate-500">
          {result.sources.map((s, i) => sourceLabel(s, i)).join(" · ")}
        </span>
      )}
    </div>
  );
}
