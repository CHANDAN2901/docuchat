# Deploy docuchat: Vercel + Render + Supabase

The Next.js frontend runs on Vercel, FastAPI runs on Render, a Supabase
PostgreSQL project stores accounts, and a Qdrant Cloud cluster stores
document chunks and embeddings. One GitHub repository is enough. This
guide describes setup; it does not mean the application has been deployed.

```text
Browser → https://YOUR-FRONTEND.vercel.app
                /api/* → https://YOUR-BACKEND.onrender.com/api/*
                                    → Supabase (accounts) / Qdrant Cloud (documents) / OpenAI (the visitor's own key)
```

This follows `week-06-profilefit/docs/deployment.md`'s already-proven
version of this same split almost exactly. The differences: this app has
no Google OAuth (plain email+password instead), and it needs a Qdrant
Cloud cluster in addition to Postgres — Postgres here only ever holds the
`users` and `login_sessions` tables, never document content.

## 1. Push the project source

Commit the source, `frontend/package-lock.json`, `backend/uv.lock`, and
the Alembic migrations. Keep `.env`/`.env.local` out of Git — the
project's `.gitignore` already excludes them except the `.example` files.

## 2. Create the Qdrant Cloud cluster

Reuse an existing Qdrant Cloud cluster or create a free one at
[cloud.qdrant.io](https://cloud.qdrant.io). Use a **new collection name**
(e.g. `docuchat_chunks`) — don't point this at a collection that already
holds data from another project, since the owner-filtering logic assumes
every point in the collection was ingested through this app. Note: free
clusters auto-pause after inactivity; the first request after idle time
may see a connection reset until it resumes from the dashboard.

## 3. Configure the Render backend

Create a **Python Web Service** from the repository, root directory
`backend`. Render provides `uv` automatically when the root contains
`uv.lock`.

Build command:

```bash
uv sync --locked --no-dev
```

Start command (migrates before starting — fine for a single instance):

```bash
.venv/bin/alembic upgrade head && exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
```

Set the health check path to `/api/health`.

Environment variables:

```dotenv
ENVIRONMENT=production
PUBLIC_URL=https://YOUR-FRONTEND.vercel.app
DATABASE_URL=YOUR_SUPABASE_POSTGRESQL_PSYCOPG_URL
QDRANT_URL=https://YOUR-CLUSTER.cloud.qdrant.io
QDRANT_API_KEY=YOUR_QDRANT_API_KEY
QDRANT_COLLECTION=docuchat_chunks
OPENAI_MODEL=gpt-4.1-mini
```

Copy the Supabase connection string with its `postgresql+psycopg://` prefix
and `sslmode=require` (use the **Session pooler** URI from Supabase's
Connect panel). **No `OPENAI_API_KEY` here** — every user supplies their
own from the browser (BYOK); the backend never stores one server-side.

## 4. Configure the Vercel frontend

Import the repository, root directory `frontend`, framework preset
**Next.js**, Node.js 22.x.

```dotenv
API_INTERNAL_URL=https://YOUR-BACKEND.onrender.com
NEXT_PUBLIC_SITE_URL=https://YOUR-FRONTEND.vercel.app
```

`API_INTERNAL_URL` must be Render's **public HTTPS URL** — Vercel cannot
reach Render's private network. `next.config.ts`'s `rewrites()` proxies
`/api/*`, including auth and uploads, to the backend, which is what makes
the session cookie same-origin and CORS unnecessary. These values must be
present at build time; redeploy after changing them.

## 5. Verify the live deployment

1. Open the Render URL at `/api/health`, then the Vercel URL at
   `/api/health`. Both should return `{"status":"ok"}`.
2. Open the Vercel URL, sign up with a real email+password, confirm you
   land on the dashboard.
3. Add your own OpenAI key, upload a PDF, ask it a question, confirm a
   grounded answer with citations naming the right document and page.
4. Sign up a **second** account in a private/incognito window, confirm its
   document list is empty and it cannot see the first account's uploads —
   this is the one property that must never regress.
5. Log out, confirm `/api/auth/me` now returns 401.

If requests return **Invalid host header**, check `trusted_hosts` logic in
`backend/app/config.py` against the actual Render hostname — add it via
`EXTRA_ALLOWED_HOSTS` if Render's `RENDER_EXTERNAL_HOSTNAME` isn't picked
up automatically. If a mutating request returns **Request origin is not
allowed**, check `PUBLIC_URL` against the browser's actual production
origin. If proxy requests fail with `DNS_HOSTNAME_RESOLVED_PRIVATE`, the
rewrite reached a private address — confirm `API_INTERNAL_URL` is the
public HTTPS Render origin and redeploy Vercel.

Render's free tier can sleep when idle — the first request after a cold
start will be slow (loading the local embedding model) but should still
succeed.
