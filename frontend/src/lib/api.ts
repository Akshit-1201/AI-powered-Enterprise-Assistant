// Thin typed client for the FastAPI backend. One request() helper attaches the Bearer
// token and normalizes errors into ApiError(status, message) so the UI can react (e.g.
// 401 -> log out, 503 -> "AI service unavailable").
import { getToken } from "./auth";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

// FastAPI returns `detail` as a string for app errors (e.g. 401/409/415/503) but as a
// list of {loc, msg} objects for 422 validation errors. Normalize both to a clean string.
interface ValidationItem {
  loc?: Array<string | number>;
  msg?: string;
}

export function extractErrorMessage(data: unknown, status: number): string {
  const detail = (data as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const msgs = (detail as ValidationItem[])
      .map((e) => {
        const field = e.loc?.[e.loc.length - 1];
        const msg = e.msg ?? "Invalid value";
        // Drop the leading "body"/"query" segment; show the offending field if present.
        return field !== undefined && field !== "body" ? `${field}: ${msg}` : msg;
      })
      .filter(Boolean);
    if (msgs.length > 0) return msgs.join("; ");
  }
  return `Request failed (${status}).`;
}

// Mirrors the backend response shapes.
export interface AskResponse {
  answer: string;
  intent: string; // action | knowledge | general | blocked
  tool_used: string | null;
  sources: Array<Record<string, unknown>>;
  session_id: string;
}

// /ask/stream metadata, delivered as the final SSE event after the answer tokens.
export interface StreamMeta {
  intent: string;
  tool_used: string | null;
  sources: Array<Record<string, unknown>>;
  session_id: string;
}

export interface StreamHandlers {
  onToken: (text: string) => void;
  onMeta: (meta: StreamMeta) => void;
  onError?: (detail: string) => void;
  onDone?: () => void;
}

export interface DocumentOut {
  id: number;
  filename: string;
  chunk_count: number;
  uploaded_at: string;
}

// Saved chat history.
export interface Conversation {
  session_id: string;
  title: string | null;
  created_at: string;
  last_active: string;
}

export interface ConversationMessage {
  role: "user" | "assistant";
  text: string;
}

export interface ConversationDetail {
  session_id: string;
  title: string | null;
  messages: ConversationMessage[];
}

export interface UploadResponse {
  id: number;
  filename: string;
  chunk_count: number;
  message: string;
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  withAuth = true,
): Promise<T> {
  const headers = new Headers(init.headers);
  if (withAuth) {
    const token = getToken();
    if (token) headers.set("Authorization", `Bearer ${token}`);
  }

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  } catch {
    // Network/CORS failure — backend down or wrong base URL.
    throw new ApiError(0, "Cannot reach the backend. Is it running on " + API_BASE + "?");
  }

  // 204 / empty body tolerance.
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;

  if (!res.ok) {
    throw new ApiError(res.status, extractErrorMessage(data, res.status));
  }
  return data as T;
}

export const api = {
  register(email: string, password: string) {
    return request<{ id: number; email: string; created_at: string }>(
      "/auth/register",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      },
      false,
    );
  },

  login(email: string, password: string) {
    return request<{ access_token: string; token_type: string }>(
      "/auth/login",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      },
      false,
    );
  },

  ask(question: string, sessionId: string | null) {
    return request<AskResponse>("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        ...(sessionId ? { session_id: sessionId } : {}),
      }),
    });
  },

  // Streaming variant of ask(). Uses fetch + a body reader (not EventSource, which can't
  // send the Authorization header) and parses the SSE frames, dispatching to callbacks.
  async askStream(
    question: string,
    sessionId: string | null,
    handlers: StreamHandlers,
  ): Promise<void> {
    const headers = new Headers({ "Content-Type": "application/json" });
    const token = getToken();
    if (token) headers.set("Authorization", `Bearer ${token}`);

    let res: Response;
    try {
      res = await fetch(`${API_BASE}/ask/stream`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          question,
          ...(sessionId ? { session_id: sessionId } : {}),
        }),
      });
    } catch {
      throw new ApiError(0, "Cannot reach the backend. Is it running on " + API_BASE + "?");
    }

    // Auth/validation failures happen before streaming starts -> normal error body.
    if (!res.ok || !res.body) {
      const text = await res.text();
      const data = text ? JSON.parse(text) : null;
      throw new ApiError(res.status, extractErrorMessage(data, res.status));
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      for (;;) {
        const sep = buffer.indexOf("\n\n");
        if (sep === -1) break;
        const frame = buffer.slice(0, sep).trim();
        buffer = buffer.slice(sep + 2);
        if (!frame.startsWith("data:")) continue;
        const json = frame.slice(5).trim();
        if (!json) continue;

        let evt: { type?: string; [k: string]: unknown };
        try {
          evt = JSON.parse(json);
        } catch {
          continue;
        }
        if (evt.type === "token") handlers.onToken(String(evt.text ?? ""));
        else if (evt.type === "meta") handlers.onMeta(evt as unknown as StreamMeta);
        else if (evt.type === "error") handlers.onError?.(String(evt.detail ?? "Something went wrong."));
        else if (evt.type === "done") handlers.onDone?.();
      }
    }
  },

  listDocuments() {
    return request<DocumentOut[]>("/documents", { method: "GET" });
  },

  deleteDocument(id: number) {
    return request<null>(`/documents/${id}`, { method: "DELETE" });
  },

  listConversations() {
    return request<Conversation[]>("/conversations", { method: "GET" });
  },

  getConversation(sessionId: string) {
    return request<ConversationDetail>(
      `/conversations/${encodeURIComponent(sessionId)}`,
      { method: "GET" },
    );
  },

  deleteConversation(sessionId: string) {
    return request<null>(`/conversations/${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    });
  },

  uploadDocument(file: File) {
    const form = new FormData();
    form.append("file", file);
    // No Content-Type header: the browser sets the multipart boundary.
    return request<UploadResponse>("/documents/upload", {
      method: "POST",
      body: form,
    });
  },
};
