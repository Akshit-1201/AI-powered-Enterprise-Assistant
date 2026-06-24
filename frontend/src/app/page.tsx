"use client";

import { useEffect, useState } from "react";
import { AuthPanel } from "@/components/AuthPanel";
import { Chat } from "@/components/Chat";
import { ConversationsPanel } from "@/components/ConversationsPanel";
import { UploadPanel } from "@/components/UploadPanel";
import { clearSession, getEmail, getToken } from "@/lib/auth";

export default function Home() {
  // null = still resolving sessionStorage on the client (avoids an auth-screen flash).
  const [authed, setAuthed] = useState<boolean | null>(null);
  const [email, setEmail] = useState<string>("");
  // Active chat (null = new chat) and a counter to refresh the chat list after a turn.
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [histVersion, setHistVersion] = useState(0);

  useEffect(() => {
    setAuthed(Boolean(getToken()));
    setEmail(getEmail() ?? "");
  }, []);

  function logout() {
    clearSession();
    setAuthed(false);
    setEmail("");
    setActiveSessionId(null); // don't carry one user's chat into the next session
  }

  if (authed === null) {
    return <div className="min-h-screen" />; // brief, blank while resolving
  }

  if (!authed) {
    return (
      <AuthPanel
        onAuthed={(e) => {
          setEmail(e);
          setAuthed(true);
        }}
      />
    );
  }

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-3">
        <div>
          <h1 className="text-sm font-semibold">Enterprise AI Assistant</h1>
          <p className="text-xs text-slate-400">{email}</p>
        </div>
        <button
          onClick={logout}
          className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-600 transition hover:border-ink hover:text-ink"
        >
          Sign out
        </button>
      </header>

      <main className="grid flex-1 grid-cols-1 overflow-hidden md:grid-cols-[240px_1fr_320px]">
        <aside className="order-3 overflow-hidden border-t border-slate-200 bg-slate-50 md:order-1 md:border-r md:border-t-0">
          <ConversationsPanel
            activeSessionId={activeSessionId}
            refreshSignal={histVersion}
            onSelect={setActiveSessionId}
            onNew={() => setActiveSessionId(null)}
            onActiveDeleted={() => setActiveSessionId(null)}
            onUnauthorized={logout}
          />
        </aside>
        <section className="order-1 overflow-hidden md:order-2">
          <Chat
            onUnauthorized={logout}
            sessionId={activeSessionId}
            onSessionId={setActiveSessionId}
            onSaved={() => setHistVersion((v) => v + 1)}
          />
        </section>
        <aside className="order-2 overflow-hidden border-b border-slate-200 bg-slate-50 md:order-3 md:border-b-0 md:border-l">
          <UploadPanel onUnauthorized={logout} />
        </aside>
      </main>
    </div>
  );
}
