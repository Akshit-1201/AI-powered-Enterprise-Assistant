// Client-side token storage. localStorage per plan D11 — simple for the build
// challenge; the XSS tradeoff (httpOnly cookie is the production answer) is a deliberate,
// documented choice. SSR-guarded so it never touches window on the server.

const TOKEN_KEY = "eai_token";
const EMAIL_KEY = "eai_email";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function getEmail(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(EMAIL_KEY);
}

export function setSession(token: string, email: string): void {
  window.localStorage.setItem(TOKEN_KEY, token);
  window.localStorage.setItem(EMAIL_KEY, email);
}

export function clearSession(): void {
  window.localStorage.removeItem(TOKEN_KEY);
  window.localStorage.removeItem(EMAIL_KEY);
}
