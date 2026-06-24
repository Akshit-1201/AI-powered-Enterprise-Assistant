"use client";

import { useCallback, useEffect, useState } from "react";
import { api, ApiError, type Conversation } from "@/lib/api";

export function ConversationsPanel({
  activeSessionId,
  refreshSignal,
  onSelect,
  onNew,
  onActiveDeleted,
  onUnauthorized,
}: {
  activeSessionId: string | null;
  refreshSignal: number; // bump to refetch (e.g. after a turn is saved)
  onSelect: (sessionId: string) => void;
  onNew: () => void;
  onActiveDeleted: () => void; // the currently-open chat was deleted → reset to a new chat
  onUnauthorized: () => void;
}) {
  const [items, setItems] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setItems(await api.listConversations());
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) onUnauthorized();
    } finally {
      setLoading(false);
    }
  }, [onUnauthorized]);

  useEffect(() => {
    refresh();
  }, [refresh, refreshSignal]);

  async function onDelete(sessionId: string) {
    setDeletingId(sessionId);
    try {
      await api.deleteConversation(sessionId);
      setItems((prev) => prev.filter((c) => c.session_id !== sessionId));
      if (sessionId === activeSessionId) onActiveDeleted();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) onUnauthorized();
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <div className="flex h-full flex-col p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-700">Chats</h2>
        <button
          onClick={onNew}
          className="rounded-lg bg-ink px-2.5 py-1 text-xs font-medium text-white transition hover:bg-slate-700"
        >
          + New
        </button>
      </div>

      <div className="mt-3 flex-1 overflow-y-auto">
        {loading ? (
          <p className="text-xs text-slate-400">Loading…</p>
        ) : items.length === 0 ? (
          <p className="text-xs text-slate-400">No saved chats yet.</p>
        ) : (
          <ul className="space-y-1">
            {items.map((c) => (
              <li
                key={c.session_id}
                className={
                  "group flex items-center gap-1 rounded-lg pr-1 transition " +
                  (c.session_id === activeSessionId
                    ? "bg-white ring-1 ring-slate-200"
                    : "hover:bg-white")
                }
              >
                <button
                  onClick={() => onSelect(c.session_id)}
                  className={
                    "min-w-0 flex-1 truncate px-3 py-2 text-left text-xs transition " +
                    (c.session_id === activeSessionId
                      ? "font-medium text-ink"
                      : "text-slate-600 group-hover:text-ink")
                  }
                  title={c.title ?? c.session_id}
                >
                  {c.title ?? "Untitled chat"}
                  <span className="mt-0.5 block text-[10px] font-normal text-slate-400">
                    {new Date(c.last_active).toLocaleString()}
                  </span>
                </button>
                <button
                  onClick={() => onDelete(c.session_id)}
                  disabled={deletingId === c.session_id}
                  aria-label={`Delete chat: ${c.title ?? "untitled"}`}
                  title="Delete chat"
                  className="shrink-0 rounded-md px-2 py-1 text-xs leading-none text-slate-300 transition hover:bg-rose-50 hover:text-rose-600 disabled:opacity-50"
                >
                  {deletingId === c.session_id ? "…" : "✕"}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
