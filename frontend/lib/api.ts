export type User = { id: string; email: string };
export type Citation = { source_doc: string; page_number: number; relevant_quote: string; verified: boolean };
export type UploadResponse = { doc_id: string; chunks: number };

type AskEvent =
  | { type: "delta"; text: string }
  | { type: "citations"; citations: Citation[] }
  | { type: "error"; message: string };

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

type Detail = string | { msg?: string }[] | undefined;

function readDetail(detail: Detail, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg).filter(Boolean).join("; ") || fallback;
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const { headers, ...rest } = init ?? {};
  const res = await fetch(path, {
    credentials: "include",
    headers: { "Content-Type": "application/json", ...(headers ?? {}) },
    ...rest,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}) as { detail?: Detail });
    throw new ApiError(res.status, readDetail(body.detail, `Request failed (${res.status})`));
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export const api = {
  signup: (email: string, password: string) =>
    request<User>("/api/auth/signup", { method: "POST", body: JSON.stringify({ email, password }) }),
  login: (email: string, password: string) =>
    request<User>("/api/auth/login", { method: "POST", body: JSON.stringify({ email, password }) }),
  logout: () => request<void>("/api/auth/logout", { method: "POST" }),
  me: () => request<User>("/api/auth/me"),
  documents: () => request<{ doc_ids: string[] }>("/api/documents"),

  async upload(file: File, apiKey: string): Promise<UploadResponse> {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/upload", {
      method: "POST",
      credentials: "include",
      headers: { "X-OpenAI-Key": apiKey },
      body: form,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}) as { detail?: Detail });
      throw new ApiError(res.status, readDetail(body.detail, `Upload failed (${res.status})`));
    }
    return res.json();
  },

  // Reads a text/event-stream response token-by-token. Not EventSource (which
  // can't send a POST body or a custom header) — a manual reader over the
  // same fetch() this file already uses everywhere else.
  async askStream(
    question: string,
    apiKey: string,
    docId: string | null,
    history: { question: string; answer: string }[],
    handlers: { onDelta: (text: string) => void; onCitations: (citations: Citation[]) => void },
  ): Promise<void> {
    const res = await fetch("/api/ask", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json", "X-OpenAI-Key": apiKey },
      body: JSON.stringify({ question, doc_id: docId, history }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}) as { detail?: Detail });
      throw new ApiError(res.status, readDetail(body.detail, `Request failed (${res.status})`));
    }
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";
      for (const part of parts) {
        if (!part.startsWith("data: ")) continue;
        const event: AskEvent = JSON.parse(part.slice("data: ".length));
        if (event.type === "delta") handlers.onDelta(event.text);
        else if (event.type === "citations") handlers.onCitations(event.citations);
        else throw new ApiError(502, event.message);
      }
    }
  },
};
