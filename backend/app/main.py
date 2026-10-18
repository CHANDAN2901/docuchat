import json
import re
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.vectorstores import VectorStore
from openai import AuthenticationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.auth import current_user
from app.auth import router as auth_router
from app.chain import extract_citations, ingest_pdf, list_doc_ids, new_store, retrieve_and_rerank, stream_answer
from app.config import settings
from app.models import User
from app.schemas import AskRequest, CitationOut, DocumentsResponse, UploadResponse

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_SLUG_RE = re.compile(r"[^a-z0-9]+")

app = FastAPI(title="docuchat API", version="0.1.0")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings().trusted_hosts)
app.include_router(auth_router)


@lru_cache
def get_store() -> VectorStore:
    # Lazy, not module-level-eager: connecting at import time would hit a
    # real (or unconfigured) Qdrant Cloud URL before tests get a chance to
    # swap this out for an in-memory store.
    return new_store()


@app.middleware("http")
async def security(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.url.path != "/api/auth/logout":
        origin = request.headers.get("origin")
        if origin is not None and origin != settings().public_url:
            return JSONResponse({"detail": "Request origin is not allowed."}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.get("/api/health")
def health():
    return {"status": "ok"}


def _slugify(filename: str) -> str:
    stem = Path(filename).stem.lower()
    slug = _SLUG_RE.sub("-", stem).strip("-")
    return slug or "document"


def _unique_doc_id(base: str, owner_id: str) -> str:
    existing = set(list_doc_ids(get_store().client, settings().qdrant_collection, owner_id))
    if base not in existing:
        return base
    suffix = 2
    while f"{base}-{suffix}" in existing:
        suffix += 1
    return f"{base}-{suffix}"


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile, user: User = Depends(current_user)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is too large (20MB limit).")
    doc_id = _unique_doc_id(_slugify(file.filename), user.id)
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(content)
        tmp.flush()
        chunks = ingest_pdf(Path(tmp.name), doc_id, user.id, get_store())
    if chunks == 0:
        raise HTTPException(400, "No extractable text found in that PDF.")
    return UploadResponse(doc_id=doc_id, chunks=chunks)


@app.get("/api/documents", response_model=DocumentsResponse)
def documents(user: User = Depends(current_user)):
    return DocumentsResponse(doc_ids=list_doc_ids(get_store().client, settings().qdrant_collection, user.id))


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _stream_ask_events(
    question: str, doc_id: str | None, top_k: int, owner_id: str, api_key: str, history: list[tuple[str, str]]
):
    try:
        docs = retrieve_and_rerank(question, get_store(), api_key, owner_id, top_k=top_k, doc_id=doc_id)
        answer = ""
        for delta in stream_answer(question, docs, api_key, history=history):
            answer += delta
            yield _sse({"type": "delta", "text": delta})
        citations = extract_citations(question, docs, answer, api_key)
        yield _sse(
            {
                "type": "citations",
                "citations": [
                    CitationOut(
                        source_doc=c.source_doc, page_number=c.page_number, relevant_quote=c.relevant_quote, verified=c.verified
                    ).model_dump()
                    for c in citations
                ],
            }
        )
    except AuthenticationError:
        yield _sse({"type": "error", "message": "That OpenAI API key was rejected. Check it and try again."})


@app.post("/api/ask")
def ask(body: AskRequest, user: User = Depends(current_user), x_openai_key: str | None = Header(default=None)):
    if not x_openai_key:
        raise HTTPException(400, "Missing OpenAI API key — add yours in settings.")
    history = [(h.question, h.answer) for h in body.history]
    return StreamingResponse(
        _stream_ask_events(body.question, body.doc_id, body.top_k, user.id, x_openai_key, history),
        media_type="text/event-stream",
    )
