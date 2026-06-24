"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { api, ApiError, type DocumentOut } from "@/lib/api";

export function UploadPanel({ onUnauthorized }: { onUnauthorized: () => void }) {
  const [docs, setDocs] = useState<DocumentOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  function handle401(err: unknown): boolean {
    if (err instanceof ApiError && err.status === 401) {
      onUnauthorized();
      return true;
    }
    return false;
  }

  async function refresh() {
    setError(null);
    try {
      setDocs(await api.listDocuments());
    } catch (err) {
      if (handle401(err)) return;
      setError(err instanceof ApiError ? err.message : "Could not load documents.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setNotice(null);
    setBusy(true);
    try {
      const res = await api.uploadDocument(file);
      setNotice(`Indexed “${res.filename}” into ${res.chunk_count} chunk${res.chunk_count > 1 ? "s" : ""}.`);
      await refresh();
    } catch (err) {
      if (handle401(err)) return;
      setError(err instanceof ApiError ? err.message : "Upload failed.");
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  return (
    <div className="flex h-full flex-col p-4">
      <h2 className="text-sm font-semibold text-slate-700">Your documents</h2>
      <p className="mt-1 text-xs text-slate-500">
        Upload PDF, .txt, or .md (max 10 MB). Knowledge questions are grounded in these.
      </p>

      <label className="mt-3 block">
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.txt,.md"
          onChange={onUpload}
          disabled={busy}
          className="block w-full text-xs text-slate-500 file:mr-3 file:cursor-pointer file:rounded-lg file:border-0 file:bg-ink file:px-3 file:py-2 file:text-xs file:font-medium file:text-white hover:file:bg-slate-700 disabled:opacity-50"
        />
      </label>

      {busy && <p className="mt-2 text-xs text-slate-400">Uploading & indexing…</p>}
      {notice && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-2 rounded-lg bg-emerald-50 px-3 py-2 text-xs text-emerald-700"
        >
          {notice}
        </motion.p>
      )}
      {error && (
        <p className="mt-2 rounded-lg bg-rose-50 px-3 py-2 text-xs text-rose-700">{error}</p>
      )}

      <div className="mt-4 flex-1 overflow-y-auto">
        {loading ? (
          <p className="text-xs text-slate-400">Loading…</p>
        ) : docs.length === 0 ? (
          <p className="text-xs text-slate-400">No documents yet.</p>
        ) : (
          <ul className="space-y-2">
            {docs.map((d) => (
              <li
                key={d.id}
                className="rounded-lg bg-white px-3 py-2 text-xs ring-1 ring-slate-200"
              >
                <p className="truncate font-medium text-slate-700">{d.filename}</p>
                <p className="text-slate-400">
                  {d.chunk_count} chunk{d.chunk_count > 1 ? "s" : ""} ·{" "}
                  {new Date(d.uploaded_at).toLocaleString()}
                </p>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
