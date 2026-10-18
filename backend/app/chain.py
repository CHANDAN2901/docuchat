"""The RAG engine: extract -> chunk -> embed -> store -> retrieve -> cite,
hybrid search (dense + sparse) with LLM re-ranking and contextual chunk
prefixes, ported from a learning exercise
(ai-workbench/exercises/week-08-hybrid-search) where each piece was built
and live-verified one at a time (95% pass rate on a 20-question eval).

Two things changed for product use, both additive:
- `owner_id` is now a required argument on every ingest/retrieve call and is
  folded into a Qdrant payload filter, so one user's chunks are never
  visible to another's query — not a convention, structurally enforced.
- `_llm()` takes the caller's own OpenAI API key per call (bring-your-own-key)
  instead of a single server-side key, since this app never stores a user's
  OpenAI key.
"""

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from zlib import crc32

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings, SparseVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from app.config import settings

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# ponytail: fixed hashing-trick vocab size for the sparse vectors below —
# collisions are rare at this size and harmless for keyword-overlap scoring.
_SPARSE_DIM = 1 << 20
_TOKEN_RE = re.compile(r"\w+")
# A short, mostly-uppercase line — "CHAPTER I PRELIMINARY" style running
# headers. A wrong "header" is just a slightly-off prefix, not corrupted
# content, so a few false positives on an unusual document are harmless.
_HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 ,\-&]{3,80}$")
# Reserved token budget for the "doc_id — section header" prefix added
# below, so the full prefixed text still fits under the embedding model's
# token limit instead of silently truncating it.
_CONTEXT_BUDGET = 30

_ANSWER_SYSTEM_PROMPT = (
    "You're continuing an ongoing conversation about the user's documents — earlier turns "
    "are included below for context, so resolve references like 'that' or 'the previous one' "
    "against them. Answer only using the numbered context for any new facts, in plain prose. "
    "If the context doesn't answer the question, say so. "
    "Never mention context block numbers or write bracketed references like [1] — "
    "sources are shown to the reader separately."
)
_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _ANSWER_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("user", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)
_MAX_HISTORY_TURNS = 6  # ponytail: fixed cap, not token-measured — generous for a Q&A tool, revisit if conversations run long

# A separate, non-streamed call extracts citations against the answer that was
# actually shown — structured-output streaming was tried and measured first
# (verified live: OpenAI + with_structured_output().stream() yields exactly one
# chunk, the complete object, not incremental text) rather than assumed to work,
# so token-by-token streaming needs a plain (non-structured) call instead.
_CITATIONS_SYSTEM_PROMPT = (
    "Given the context blocks and the answer already given, list one citation entry "
    "for every distinct fact or claim in the answer that comes from the context, naming "
    "the block it came from and the exact supporting quote, copied verbatim. Leave the "
    "list empty if the answer used none of the context."
)
_CITATIONS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _CITATIONS_SYSTEM_PROMPT),
        ("user", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:\n{answer}"),
    ]
)

_RERANK_SYSTEM_PROMPT = (
    "Rank the numbered context blocks by how relevant each is to answering the "
    "question. Return their indices only, most relevant first."
)
_RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _RERANK_SYSTEM_PROMPT),
        ("user", "Context blocks:\n{context}\n\nQuestion: {question}\n\nReturn the top {top_k} indices."),
    ]
)


class RerankResult(BaseModel):
    ranked_indices: list[int] = Field(description="Block indices ordered most-relevant first.")


class RawCitation(BaseModel):
    source_index: int = Field(description="The [N] number of the context block this citation is drawn from.")
    quote: str = Field(description="The exact supporting quote, copied verbatim from that context block.")


class CitationsResult(BaseModel):
    citations: list[RawCitation]


@dataclass
class Citation:
    source_doc: str
    page_number: int
    relevant_quote: str
    verified: bool


class KeywordSparseEmbeddings(SparseEmbeddings):
    """Term-frequency sparse vectors via the hashing trick (each token -> a
    stable index via crc32, value = count in that text). No ML model, no
    ONNX runtime. Qdrant's server-side `Modifier.IDF` (set on the collection
    below) turns these raw counts into real BM25-style scoring — the IDF
    half of BM25 doesn't need a model, just document frequency stats Qdrant
    already tracks internally.
    """

    def _sparse_vector(self, text: str) -> SparseVector:
        counts = Counter(crc32(token.encode()) % _SPARSE_DIM for token in _TOKEN_RE.findall(text.lower()))
        if not counts:
            return SparseVector(indices=[], values=[])
        indices, values = zip(*counts.items())
        return SparseVector(indices=list(indices), values=[float(v) for v in values])

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._sparse_vector(text) for text in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._sparse_vector(text)


@lru_cache
def _embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache
def _embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@lru_cache
def _sparse_embeddings() -> KeywordSparseEmbeddings:
    return KeywordSparseEmbeddings()


@lru_cache
def _splitter() -> RecursiveCharacterTextSplitter:
    # -2 for the [CLS]/[SEP] special tokens. -_CONTEXT_BUDGET to leave room
    # for the doc_id/section-header prefix ingest_pdf() adds before embedding.
    chunk_size = _embedding_model().max_seq_length - 2 - _CONTEXT_BUDGET
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        _embedding_model().tokenizer, chunk_size=chunk_size, chunk_overlap=20
    )


def _section_headers(pages: list[Document]) -> list[str]:
    """One running section header per page — a page without its own heading
    inherits the most recent one seen, since legal/technical PDFs commonly
    repeat the enclosing chapter/section name as a running header on every
    page within it. Empty string if no heading has been seen yet."""
    headers = []
    current = ""
    for page in pages:
        lines = page.page_content.strip().splitlines()
        first_line = lines[0].strip() if lines else ""
        if _HEADER_RE.match(first_line):
            current = first_line
        headers.append(current)
    return headers


@lru_cache
def _llm(api_key: str) -> ChatOpenAI:
    """Keyed on the caller's own OpenAI key (bring-your-own-key) — never a
    server-side key. `lru_cache`'s default maxsize is plenty for a handful
    of distinct users reusing the same client across requests."""
    return ChatOpenAI(model=settings().openai_model, api_key=api_key)


def new_store(client_options: dict | None = None) -> VectorStore:
    """Dense (MiniLM) + sparse (keyword) hybrid search. Qdrant fuses the two
    independently-searched result sets natively (RRF by default) rather than
    reranking a vector-only shortlist.

    `client_options` defaults to the live Qdrant Cloud cluster; tests pass
    `{"location": ":memory:"}` instead.
    """
    cfg = settings()
    return QdrantVectorStore.construct_instance(
        embedding=_embeddings(),
        sparse_embedding=_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        client_options=client_options or {"url": cfg.qdrant_url, "api_key": cfg.qdrant_api_key},
        collection_name=cfg.qdrant_collection,
        sparse_vector_params={"modifier": models.Modifier.IDF},
    )


def _owner_filter(owner_id: int, doc_id: str | None = None) -> models.Filter:
    must = [models.FieldCondition(key="metadata.owner_id", match=models.MatchValue(value=owner_id))]
    if doc_id is not None:
        must.append(models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value=doc_id)))
    return models.Filter(must=must)


def ingest_pdf(path: Path, doc_id: str, owner_id: int, store: VectorStore) -> int:
    """`split_documents` splits each page's text independently, so
    `chunk_overlap` never bridges a page break — a heading at the bottom of
    a page can land in a different chunk than its own following content.
    Fixed by seeding each page's text with a tail slice of the previous
    page's before splitting — same effect as `chunk_overlap`, just reaching
    across the page boundary it doesn't cross.
    """
    pages = PyMuPDFLoader(str(path)).load()
    headers = _section_headers(pages)
    for i in range(1, len(pages)):
        # ponytail: fixed 200-char bridge, not token-measured — good enough to
        # carry a trailing heading across the page break.
        tail = pages[i - 1].page_content[-200:]
        pages[i].page_content = f"{tail}\n{pages[i].page_content}"
    splits = _splitter().split_documents(pages)
    if not splits:
        return 0
    for split in splits:
        split.metadata["doc_id"] = doc_id
        split.metadata["owner_id"] = owner_id
        header = headers[split.metadata.get("page", 0)]
        context = f"{doc_id} — {header}" if header else doc_id
        split.page_content = f"{context}\n{split.page_content}"
    store.add_documents(splits)
    return len(splits)


def list_doc_ids(client: QdrantClient, collection: str, owner_id: int) -> list[str]:
    """Distinct `doc_id`s this owner has ingested — scoped by the same
    owner filter every retrieval call uses, so this never leaks another
    user's document names either."""
    doc_ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=_owner_filter(owner_id),
            limit=1000,
            offset=offset,
            with_payload=True,
        )
        doc_ids.update(p.payload["metadata"]["doc_id"] for p in points)
        if offset is None:
            break
    return sorted(doc_ids)


def _build_context(docs: list[Document]) -> str:
    blocks = [
        f"[{i}] (source: {doc.metadata['doc_id']}, page {doc.metadata.get('page', 0) + 1})\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    ]
    return "\n\n".join(blocks)


def _build_citations(raw_citations: list[RawCitation], docs: list[Document]) -> list[Citation]:
    citations = []
    for raw in raw_citations:
        index = raw.source_index - 1
        if not 0 <= index < len(docs):
            continue
        doc = docs[index]
        citations.append(
            Citation(
                source_doc=doc.metadata["doc_id"],
                page_number=doc.metadata.get("page", 0) + 1,
                relevant_quote=raw.quote,
                verified=raw.quote.strip() in doc.page_content,
            )
        )
    return citations


def _rerank(question: str, docs: list[Document], top_k: int, api_key: str) -> list[Document]:
    if len(docs) <= top_k:
        return docs
    messages = _RERANK_PROMPT.invoke({"context": _build_context(docs), "question": question, "top_k": top_k})
    result: RerankResult = _llm(api_key).with_structured_output(RerankResult).invoke(messages)
    seen: set[int] = set()
    selected = []
    for index in result.ranked_indices:
        position = index - 1
        if 0 <= position < len(docs) and position not in seen:
            seen.add(position)
            selected.append(docs[position])
        if len(selected) == top_k:
            break
    return selected or docs[:top_k]  # ponytail: fall back to retrieval order if the model returns nothing usable


def retrieve_and_rerank(
    question: str,
    store: VectorStore,
    api_key: str,
    owner_id: int,
    top_k: int = 5,
    rerank_pool: int = 40,
    doc_id: str | None = None,
) -> list[Document]:
    candidates = store.similarity_search(question, k=rerank_pool, filter=_owner_filter(owner_id, doc_id))
    return _rerank(question, candidates, top_k, api_key)


def stream_answer(
    question: str, docs: list[Document], api_key: str, history: list[tuple[str, str]] | None = None
) -> Iterator[str]:
    """Plain-text token streaming for the visible answer — citations are
    extracted separately (see `extract_citations`) since structured-output
    streaming doesn't give incremental text (verified: one chunk, not many).

    `history` is prior (question, answer) turns from the same conversation —
    without it, every question is answered as if it were the first, with no
    way to resolve "what about that" against an earlier turn.
    """
    history_messages = []
    for prior_question, prior_answer in (history or [])[-_MAX_HISTORY_TURNS:]:
        history_messages.append(HumanMessage(prior_question))
        history_messages.append(AIMessage(prior_answer))
    messages = _ANSWER_PROMPT.invoke(
        {"context": _build_context(docs), "question": question, "history": history_messages}
    )
    for chunk in _llm(api_key).stream(messages):
        if chunk.content:
            yield chunk.content


def extract_citations(question: str, docs: list[Document], answer: str, api_key: str) -> list[Citation]:
    messages = _CITATIONS_PROMPT.invoke({"context": _build_context(docs), "question": question, "answer": answer})
    result: CitationsResult = _llm(api_key).with_structured_output(CitationsResult).invoke(messages)
    return _build_citations(result.citations, docs)
