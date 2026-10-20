from pathlib import Path

from app import chain

FIXTURE = Path(__file__).parent / "fixtures" / "React_Native_Guide_Plain.pdf"


def test_upload_rejects_non_pdf(signed_up_client):
    resp = signed_up_client.post(
        "/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")}, headers={"X-OpenAI-Key": "sk-test"}
    )
    assert resp.status_code == 400


def test_upload_without_openai_key_header_returns_400(signed_up_client):
    with FIXTURE.open("rb") as f:
        resp = signed_up_client.post("/api/upload", files={"file": ("guide.pdf", f, "application/pdf")})
    assert resp.status_code == 400


def test_upload_then_list_documents(signed_up_client):
    with FIXTURE.open("rb") as f:
        resp = signed_up_client.post(
            "/api/upload", files={"file": ("react-native-guide.pdf", f, "application/pdf")}, headers={"X-OpenAI-Key": "sk-test"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "react-native-guide"
    assert body["chunks"] > 0

    docs = signed_up_client.get("/api/documents")
    assert docs.json()["doc_ids"] == ["react-native-guide"]


def test_ask_without_openai_key_header_returns_400(signed_up_client):
    resp = signed_up_client.post("/api/ask", json={"question": "What is Expo?"})
    assert resp.status_code == 400


def test_ask_streams_an_error_event_on_openai_auth_failure(signed_up_client, monkeypatch):
    def fake_retrieve_and_rerank(*args, **kwargs):
        import httpx
        from openai import AuthenticationError

        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        raise AuthenticationError("bad key", response=httpx.Response(401, request=request), body=None)

    # The HTTP status is already 200 (SSE headers sent) once a streaming
    # response starts — an error that happens mid-pipeline comes through as
    # an event in the stream, not an HTTP status code.
    monkeypatch.setattr("app.main.retrieve_and_rerank", fake_retrieve_and_rerank)

    resp = signed_up_client.post(
        "/api/ask", json={"question": "What is Expo?"}, headers={"X-OpenAI-Key": "sk-invalid"}
    )

    assert resp.status_code == 200
    assert '"type": "error"' in resp.text


def test_ask_streams_answer_deltas_then_a_citations_event(signed_up_client, monkeypatch):
    monkeypatch.setattr("app.main.retrieve_and_rerank", lambda *a, **k: ["fake-doc"])
    monkeypatch.setattr("app.main.stream_answer", lambda *a, **k: iter(["Ex", "po is a tool."]))
    monkeypatch.setattr(
        "app.main.extract_citations",
        lambda *a, **k: [chain.Citation(source_doc="guide", page_number=1, relevant_quote="Expo", verified=True)],
    )

    resp = signed_up_client.post(
        "/api/ask", json={"question": "What is Expo?"}, headers={"X-OpenAI-Key": "sk-test"}
    )

    assert resp.status_code == 200
    events = [line[len("data: ") :] for line in resp.text.splitlines() if line.startswith("data: ")]
    assert '"type": "delta", "text": "Ex"' in events[0]
    assert '"type": "delta", "text": "po is a tool."' in events[1]
    assert '"type": "citations"' in events[2]
    assert '"source_doc": "guide"' in events[2]


def test_re_uploading_the_same_filename_gets_a_deduped_doc_id(signed_up_client):
    for _ in range(2):
        with FIXTURE.open("rb") as f:
            resp = signed_up_client.post(
                "/api/upload", files={"file": ("guide.pdf", f, "application/pdf")}, headers={"X-OpenAI-Key": "sk-test"}
            )
        assert resp.status_code == 200

    doc_ids = signed_up_client.get("/api/documents").json()["doc_ids"]
    assert doc_ids == ["guide", "guide-2"]
