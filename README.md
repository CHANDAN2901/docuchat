# docuchat

Chat with your own documents. Upload a PDF, ask questions, get grounded
answers with citations naming exactly which document and page each fact
came from. Every account's documents are completely private — no one else
can see or query them, not even other users of the same deployment.

You bring your own OpenAI API key (entered once in the browser, stored only
there) — this app never charges you for anyone else's usage and never sees
or stores your key server-side.

The RAG engine (hybrid dense+sparse search, LLM re-ranking, contextual
chunk embeddings) was built and live-verified in a separate learning
exercise (`ai-workbench/exercises/week-08-hybrid-search`, 95% pass rate on
a 20-question eval) and ported here with one addition: every document and
every query is scoped to its owner, enforced in the vector store filter
itself — not a UI convention.

## Stack

- **Backend**: FastAPI + SQLAlchemy/Alembic (Postgres in production, SQLite
  for local dev) + Qdrant (hybrid vector search) — `backend/`
- **Frontend**: Next.js (App Router) + Tailwind — `frontend/`
- **Auth**: email + password, bcrypt-hashed, opaque session tokens in an
  httpOnly cookie (not JWT — trivially revocable on logout, no signing
  secret to manage)

## Local development

Backend:

```bash
cd backend
cp .env.example .env   # fill in QDRANT_URL/QDRANT_API_KEY (Qdrant Cloud,
                        # or `docker run -p 6333:6333 qdrant/qdrant` for local dev)
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000
```

Frontend (separate terminal):

```bash
cd frontend
cp .env.example .env.local   # defaults are fine for local dev
npm install
npm run dev
```

Open `http://localhost:3000`, sign up, upload a PDF, add your OpenAI key
in the dashboard, and ask a question.

## Tests

```bash
cd backend
uv run pytest
```

21 tests: the ported RAG-engine tests (hybrid search, re-ranking,
contextual chunking — all offline, no API key needed), auth tests
(signup/login/logout/duplicate-email/401-without-session), and
`test_isolation.py` — the test that actually proves the core promise: two
real signed-up users, each uploading through the real HTTP stack, and
neither can see or query the other's documents.

## Deployment

See [`docs/deployment.md`](docs/deployment.md) — Vercel (frontend) +
Render (backend) + Supabase (Postgres), the same stack already proven
working for `week-06-profilefit` in this workspace.

## What's deliberately not built yet

- Password reset / email verification — real email+password auth, but no
  "forgot password" flow yet.
- Document deletion.
- Any sharing between users — isolation is total, by design, for now.
- Rate limiting — BYOK already puts OpenAI cost risk on the visitor, not
  the deployer, and usage at small scale doesn't need it yet.
