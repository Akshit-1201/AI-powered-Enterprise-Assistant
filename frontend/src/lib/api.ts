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

export interface DocumentOut {
  id: number;
  filename: string;
  chunk_count: number;
  uploaded_at: string;
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

  listDocuments() {
    return request<DocumentOut[]>("/documents", { method: "GET" });
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
