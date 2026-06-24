// Client-side token storage. We use sessionStorage (not localStorage) so a login is
// scoped to a single browser tab and cleared when the tab closes — a fresh tab, a new
// window, or a browser restart all start logged-out, and one tab's login never leaks into
// another. (localStorage, by contrast, is shared across all tabs of a profile and persists
// across restarts.) The XSS tradeoff is the same as localStorage; the production answer is
// an httpOnly cookie. SSR-guarded so it never touches window on the server.

const TOKEN_KEY = "eai_token";
const EMAIL_KEY = "eai_email";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.sessionStorage.getItem(TOKEN_KEY);
}

export function getEmail(): string | null {
  if (typeof window === "undefined") return null;
  return window.sessionStorage.getItem(EMAIL_KEY);
}

export function setSession(token: string, email: string): void {
  window.sessionStorage.setItem(TOKEN_KEY, token);
  window.sessionStorage.setItem(EMAIL_KEY, email);
}

export function clearSession(): void {
  window.sessionStorage.removeItem(TOKEN_KEY);
  window.sessionStorage.removeItem(EMAIL_KEY);
}
