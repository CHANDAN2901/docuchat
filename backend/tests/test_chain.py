from pathlib import Path

from langchain_core.documents import Document

from app import chain

# Embedded Qdrant, no live cloud creds needed for tests.
_MEMORY_QDRANT = {"location": ":memory:"}


def test_keyword_sparse_embeddings_counts_terms_by_stable_hash():
    sparse = chain.KeywordSparseEmbeddings()

    vec = sparse.embed_query("penalty penalty filing")

    assert len(vec.indices) == 2  # "penalty" and "filing" hash to 2 distinct indices
    assert sorted(vec.values) == [1.0, 2.0]  # "penalty" x2, "filing" x1
    assert sparse.embed_query("penalty") == sparse.embed_documents(["penalty"])[0]


def test_hybrid_search_finds_exact_keyword_match_dense_search_would_miss():
    store = chain.new_store(client_options=_MEMORY_QDRANT)
    store.add_documents(
        [
            Document(page_content="The penalty for late filing is a fine per day.", metadata={"doc_id": "d", "owner_id": "u1", "page": 1}),
            Document(page_content="Bananas are a good source of potassium.", metadata={"doc_id": "d", "owner_id": "u1", "page": 2}),
            Document(page_content="Tropical fruit grows well in warm climates.", metadata={"doc_id": "d", "owner_id": "u1", "page": 3}),
        ]
    )

    docs = store.similarity_search("penalty", k=1, filter=chain._owner_filter("u1"))

    assert "penalty" in docs[0].page_content


def test_ingest_pdf_chunks_and_stores_real_pdf():
    store = chain.new_store(client_options=_MEMORY_QDRANT)

    count = chain.ingest_pdf(Path("tests/fixtures/React_Native_Guide_Plain.pdf"), "rn-guide", "u1", store)

    assert count > 0
    docs = store.similarity_search("Expo", k=1, filter=chain._owner_filter("u1"))
    assert docs[0].metadata["doc_id"] == "rn-guide"
    assert docs[0].metadata["owner_id"] == "u1"
    assert docs[0].page_content.startswith("rn-guide")  # doc_id/section-header context prefix


def test_section_headers_carries_forward_across_pages_without_their_own_heading():
    pages = [
        Document(page_content="CHAPTER I PRELIMINARY\nSome body text."),
        Document(page_content="More body text on the same chapter, no heading line."),
        Document(page_content="CHAPTER II ADMINISTRATION\nNew chapter body."),
        Document(page_content="lowercase first line, not a heading"),
    ]

    headers = chain._section_headers(pages)

    assert headers == [
        "CHAPTER I PRELIMINARY",
        "CHAPTER I PRELIMINARY",
        "CHAPTER II ADMINISTRATION",
        "CHAPTER II ADMINISTRATION",
    ]


def test_rerank_reorders_dedupes_and_drops_out_of_range_indices(monkeypatch):
    docs = [Document(page_content=f"doc {i}", metadata={"doc_id": "d", "page": 0}) for i in range(4)]

    class FakeStructuredLLM:
        def invoke(self, _input):
            return chain.RerankResult(ranked_indices=[3, 3, 1, 9])  # dup + out-of-range

    class FakeLLM:
        def with_structured_output(self, _schema):
            return FakeStructuredLLM()

    monkeypatch.setattr(chain, "_llm", lambda api_key: FakeLLM())

    reranked = chain._rerank("question", docs, top_k=2, api_key="sk-test")

    assert reranked == [docs[2], docs[0]]


def test_rerank_skips_llm_call_when_pool_not_larger_than_top_k(monkeypatch):
    def fail(api_key):
        raise AssertionError("should not call the LLM when pool <= top_k")

    monkeypatch.setattr(chain, "_llm", fail)
    docs = [Document(page_content="only one")]

    assert chain._rerank("question", docs, top_k=5, api_key="sk-test") == docs


def test_stream_answer_yields_incremental_text_chunks(monkeypatch):
    docs = [Document(page_content="Expo is a tool used in React Native development.", metadata={"doc_id": "doc", "page": 0})]

    class FakeChunk:
        def __init__(self, content):
            self.content = content

    class FakeLLM:
        def stream(self, _messages):
            yield FakeChunk("Expo ")
            yield FakeChunk("is a tool.")

    monkeypatch.setattr(chain, "_llm", lambda api_key: FakeLLM())

    chunks = list(chain.stream_answer("What is Expo?", docs, "sk-test"))

    assert chunks == ["Expo ", "is a tool."]


def test_stream_answer_passes_prior_turns_as_conversation_history(monkeypatch):
    docs = [Document(page_content="Expo is a tool used in React Native development.", metadata={"doc_id": "doc", "page": 0})]
    captured = {}

    class FakeChunk:
        def __init__(self, content):
            self.content = content

    class FakeLLM:
        def stream(self, messages):
            captured["messages"] = messages
            yield FakeChunk("because it's great.")

    monkeypatch.setattr(chain, "_llm", lambda api_key: FakeLLM())

    history = [("What is Expo?", "Expo is a tool for React Native.")]
    list(chain.stream_answer("Why use it?", docs, "sk-test", history=history))

    rendered = [m.content for m in captured["messages"].messages]
    assert any("What is Expo?" in c for c in rendered)
    assert any("Expo is a tool for React Native." in c for c in rendered)


def test_stream_answer_caps_history_to_recent_turns(monkeypatch):
    docs = [Document(page_content="irrelevant", metadata={"doc_id": "doc", "page": 0})]
    captured = {}

    class FakeChunk:
        def __init__(self, content):
            self.content = content

    class FakeLLM:
        def stream(self, messages):
            captured["messages"] = messages
            yield FakeChunk("ok")

    monkeypatch.setattr(chain, "_llm", lambda api_key: FakeLLM())

    history = [(f"question {i}", f"answer {i}") for i in range(10)]
    list(chain.stream_answer("latest question", docs, "sk-test", history=history))

    rendered = " ".join(m.content for m in captured["messages"].messages)
    assert "question 0" not in rendered  # older than the cap, dropped
    assert "question 9" in rendered  # most recent, kept


def test_extract_citations_verifies_against_real_chunk_text(monkeypatch):
    docs = [Document(page_content="Expo is a tool used in React Native development.", metadata={"doc_id": "doc", "page": 0})]

    class FakeStructuredLLM:
        def invoke(self, _input):
            return chain.CitationsResult(
                citations=[
                    chain.RawCitation(source_index=1, quote="Expo is a tool used in React Native development."),
                    chain.RawCitation(source_index=1, quote="a paraphrased quote not in the chunk"),
                    chain.RawCitation(source_index=9, quote="out of range, should be dropped"),
                ]
            )

    class FakeLLM:
        def with_structured_output(self, _schema):
            return FakeStructuredLLM()

    monkeypatch.setattr(chain, "_llm", lambda api_key: FakeLLM())

    citations = chain.extract_citations("What is Expo?", docs, "Expo is a tool for React Native.", "sk-test")

    assert len(citations) == 2  # out-of-range source_index dropped
    assert citations[0].verified is True
    assert citations[1].verified is False


def test_retrieval_never_sees_another_owners_documents():
    store = chain.new_store(client_options=_MEMORY_QDRANT)
    store.add_documents(
        [
            Document(page_content="Owner A's secret roadmap.", metadata={"doc_id": "a-doc", "owner_id": "owner-a", "page": 0}),
            Document(page_content="Owner B's unrelated notes.", metadata={"doc_id": "b-doc", "owner_id": "owner-b", "page": 0}),
        ]
    )

    candidates = store.similarity_search("roadmap", k=10, filter=chain._owner_filter("owner-b"))

    assert all(doc.metadata["owner_id"] == "owner-b" for doc in candidates)
    assert not any("secret roadmap" in doc.page_content for doc in candidates)
