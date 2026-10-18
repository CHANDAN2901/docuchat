"""The one promise this whole architecture exists to keep: a user can never
see or query another user's documents. Tested through the real HTTP stack
(two real signed-up accounts, two real cookie-bearing clients), not just at
the chain-filter level — this is what would actually leak in production if
a route forgot to pass owner_id.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from app import main as main_module

FIXTURE = Path(__file__).parent / "fixtures" / "React_Native_Guide_Plain.pdf"


def _signed_up(email: str) -> TestClient:
    client = TestClient(main_module.app)
    resp = client.post("/api/auth/signup", json={"email": email, "password": "correct-horse"})
    assert resp.status_code == 200
    return client


def test_user_b_cannot_see_user_as_uploaded_documents():
    alice = _signed_up("alice-iso@example.com")
    bob = _signed_up("bob-iso@example.com")

    with FIXTURE.open("rb") as f:
        upload = alice.post("/api/upload", files={"file": ("react-native-guide.pdf", f, "application/pdf")})
    assert upload.status_code == 200
    assert upload.json()["chunks"] > 0

    assert alice.get("/api/documents").json()["doc_ids"] == ["react-native-guide"]
    assert bob.get("/api/documents").json()["doc_ids"] == []


def test_each_signup_gets_an_independent_document_namespace():
    alice = _signed_up("alice-ns@example.com")
    bob = _signed_up("bob-ns@example.com")

    for client in (alice, bob):
        with FIXTURE.open("rb") as f:
            resp = client.post("/api/upload", files={"file": ("guide.pdf", f, "application/pdf")})
        assert resp.json()["doc_id"] == "guide"  # same filename, same slug — no cross-user collision handling needed

    assert alice.get("/api/documents").json()["doc_ids"] == ["guide"]
    assert bob.get("/api/documents").json()["doc_ids"] == ["guide"]
