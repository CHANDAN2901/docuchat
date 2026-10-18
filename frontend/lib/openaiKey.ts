const STORAGE_KEY = "docuchat_openai_key";

// Browser-only storage — this key is never sent anywhere but this app's own
// /api/ask (as a header) and is never persisted server-side.
export function getStoredKey(): string {
  if (typeof window === "undefined") return "";
  try {
    return window.localStorage.getItem(STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setStoredKey(key: string): void {
  if (typeof window === "undefined") return;
  try {
    if (key) window.localStorage.setItem(STORAGE_KEY, key);
    else window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore (private browsing / blocked storage) — the key input still works for this session
  }
}
