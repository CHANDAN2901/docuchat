"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { ApiError, type Citation, api } from "@/lib/api";
import { getStoredKey, setStoredKey } from "@/lib/openaiKey";

type Message = { question: string; answer: string; citations: Citation[] | null; streaming: boolean; error: string | null };

export default function Dashboard() {
  const router = useRouter();
  const [email, setEmail] = useState<string | null>(null);
  const [docIds, setDocIds] = useState<string[]>([]);
  const [scope, setScope] = useState<string>(""); // "" = all documents
  const [apiKey, setApiKey] = useState("");
  const [editingKey, setEditingKey] = useState(false);
  const [keyDraft, setKeyDraft] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [asking, setAsking] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api
      .me()
      .then((user) => setEmail(user.email))
      .catch(() => router.push("/login"));
    setApiKey(getStoredKey());
  }, [router]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function refreshDocuments() {
    const { doc_ids } = await api.documents();
    setDocIds(doc_ids);
  }

  useEffect(() => {
    if (email) refreshDocuments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [email]);

  function saveKey() {
    setApiKey(keyDraft);
    setStoredKey(keyDraft);
    setEditingKey(false);
  }

  async function onUpload(files: FileList | null) {
    if (!files || files.length === 0) return;
    setUploading(true);
    setUploadError(null);
    try {
      for (const file of Array.from(files)) {
        await api.upload(file);
      }
      await refreshDocuments();
    } catch (err) {
      setUploadError(err instanceof ApiError ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  // Updates the last message via the functional setState form, so each
  // update always sees the true latest state — important here since deltas
  // arrive faster than React re-renders during a token stream.
  function updateLastMessage(patch: Partial<Message> | ((m: Message) => Partial<Message>)) {
    setMessages((prev) =>
      prev.map((m, i) => (i === prev.length - 1 ? { ...m, ...(typeof patch === "function" ? patch(m) : patch) } : m)),
    );
  }

  async function onAsk(e: React.FormEvent) {
    e.preventDefault();
    if (!apiKey || !question.trim() || asking) return;
    setAsking(true);
    const asked = question;
    // Completed turns only — a failed or still-streaming one has nothing
    // settled worth feeding back in as conversation history.
    const history = messages.filter((m) => !m.streaming && !m.error).map((m) => ({ question: m.question, answer: m.answer }));
    setQuestion("");
    setMessages((prev) => [...prev, { question: asked, answer: "", citations: null, streaming: true, error: null }]);

    try {
      await api.askStream(asked, apiKey, scope || null, history, {
        onDelta: (text) => updateLastMessage((m) => ({ answer: m.answer + text })),
        onCitations: (citations) => updateLastMessage({ citations, streaming: false }),
      });
    } catch (err) {
      updateLastMessage({ streaming: false, error: err instanceof ApiError ? err.message : "Something went wrong asking that." });
    } finally {
      setAsking(false);
    }
  }

  async function onLogout() {
    await api.logout();
    router.push("/login");
  }

  if (!email) return null;

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-4 py-6 md:px-6">
      <header className="mb-6 flex items-center justify-between border-b border-line pb-4">
        <p className="text-base font-medium text-ink">docuchat</p>
        <div className="flex items-center gap-4 text-sm text-ink-soft">
          <span>{email}</span>
          <button onClick={onLogout} className="text-accent hover:underline">Log out</button>
        </div>
      </header>

      <div className="grid flex-1 gap-6 md:grid-cols-[240px_1fr]">
        <aside className="space-y-5">
          <section>
            <h2 className="mb-2 text-sm font-medium text-ink">Library</h2>
            <input
              ref={fileInputRef}
              id="upload"
              type="file"
              accept="application/pdf"
              multiple
              disabled={uploading}
              onChange={(e) => onUpload(e.target.files)}
              className="hidden"
            />
            <label
              htmlFor="upload"
              className="block cursor-pointer rounded-md border border-dashed border-line px-3 py-2 text-center text-sm text-ink-soft hover:border-accent hover:text-accent"
            >
              {uploading ? "Uploading…" : "Upload a PDF"}
            </label>
            {uploadError && <p className="mt-1.5 text-xs text-red-700">{uploadError}</p>}

            <div className="mt-3 space-y-1">
              <button
                onClick={() => setScope("")}
                className={`block w-full rounded-md px-2 py-1.5 text-left text-sm ${scope === "" ? "bg-accent-soft text-accent" : "text-ink-soft hover:bg-paper"}`}
              >
                All documents
              </button>
              {docIds.map((id) => (
                <button
                  key={id}
                  onClick={() => setScope(id)}
                  className={`block w-full truncate rounded-md px-2 py-1.5 text-left text-sm ${scope === id ? "bg-accent-soft text-accent" : "text-ink-soft hover:bg-paper"}`}
                  title={id}
                >
                  {id}
                </button>
              ))}
              {docIds.length === 0 && <p className="px-2 py-1.5 text-xs text-ink-soft">Nothing uploaded yet.</p>}
            </div>
          </section>

          <section className="border-t border-line pt-4">
            <h2 className="mb-2 text-sm font-medium text-ink">OpenAI key</h2>
            {editingKey ? (
              <div className="space-y-2">
                <input
                  type="password"
                  autoFocus
                  placeholder="sk-…"
                  value={keyDraft}
                  onChange={(e) => setKeyDraft(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && saveKey()}
                  className="w-full rounded-md border border-line bg-surface px-2 py-1.5 text-sm text-ink outline-none focus:border-accent"
                />
                <div className="flex gap-2">
                  <button onClick={saveKey} className="rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-white">Save</button>
                  <button onClick={() => setEditingKey(false)} className="text-xs text-ink-soft hover:underline">Cancel</button>
                </div>
              </div>
            ) : (
              <button
                onClick={() => {
                  setKeyDraft(apiKey);
                  setEditingKey(true);
                }}
                className="text-sm text-ink-soft hover:text-accent"
              >
                {apiKey ? "✓ Key saved — edit" : "Add your key"}
              </button>
            )}
            <p className="mt-1.5 text-xs text-ink-soft">Stored only in this browser, sent only to answer your questions.</p>
          </section>
        </aside>

        <section className="flex min-h-[60vh] flex-col">
          <div className="flex-1 space-y-6">
            {messages.length === 0 && (
              <p className="text-sm text-ink-soft">Upload a document, then ask a question grounded in what you uploaded.</p>
            )}
            {messages.map((m, i) => (
              <article key={i} className="space-y-2">
                <p className="text-sm font-medium text-ink">{m.question}</p>

                {m.streaming && m.answer === "" ? (
                  <p className="flex gap-1 text-sm text-ink-soft" aria-label="Thinking">
                    <span className="thinking-dot">●</span>
                    <span className="thinking-dot" style={{ animationDelay: "0.2s" }}>●</span>
                    <span className="thinking-dot" style={{ animationDelay: "0.4s" }}>●</span>
                  </p>
                ) : (
                  <p className={`max-w-[70ch] text-sm leading-relaxed text-ink ${m.streaming ? "streaming-caret" : ""}`}>
                    {m.answer}
                  </p>
                )}

                {m.error && <p className="text-sm text-red-700">{m.error}</p>}

                {m.citations && m.citations.length > 0 && (
                  <div className="space-y-2 pt-1">
                    {m.citations.map((c, j) => (
                      <div key={j} className="max-w-[70ch] border-l-2 border-mark bg-mark-soft py-1.5 pl-3 pr-2 text-sm">
                        <p className="font-serif italic text-ink">&ldquo;{c.relevant_quote}&rdquo;</p>
                        <p className="mt-1 text-xs text-ink-soft">
                          {c.source_doc}, page {c.page_number}
                          {!c.verified && " · paraphrased"}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </article>
            ))}
            <div ref={bottomRef} />
          </div>

          <form onSubmit={onAsk} className="sticky bottom-0 mt-4 flex gap-2 border-t border-line bg-paper pt-4">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={apiKey ? "Ask a question about your documents…" : "Add your OpenAI key first"}
              disabled={!apiKey || asking}
              className="flex-1 rounded-md border border-line bg-surface px-3 py-2 text-sm text-ink outline-none focus:border-accent disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={!apiKey || asking || !question.trim()}
              className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              Ask
            </button>
          </form>
        </section>
      </div>
    </main>
  );
}
