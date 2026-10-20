import os
import tempfile
import uuid

os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mktemp(suffix='.db')}"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

from app import chain  # noqa: E402
from app import main as main_module  # noqa: E402
from app.db import Base, engine  # noqa: E402

Base.metadata.create_all(engine())

# Real OpenAI embeddings would cost money and need network on every test
# run — a fixed-size deterministic fake keeps retrieval tests free and
# offline. Sparse (keyword) embeddings are untouched (hand-rolled, already
# free), so hybrid-search and isolation tests still exercise genuine
# keyword-match behavior; only the dense half is faked.
chain._embeddings = lambda api_key: DeterministicFakeEmbedding(size=chain._EMBEDDING_DIM)

TEST_API_KEY = "sk-test"
_test_store = chain.new_store(TEST_API_KEY, client_options={"location": ":memory:"})
main_module.get_store = lambda api_key: _test_store
main_module.get_client = lambda: _test_store.client


@pytest.fixture
def client():
    return TestClient(main_module.app)


@pytest.fixture
def signed_up_client(client):
    email = f"{uuid.uuid4()}@example.com"  # unique per test — the SQLite test DB persists across the whole session
    resp = client.post("/api/auth/signup", json={"email": email, "password": "correct-horse"})
    assert resp.status_code == 200
    return client
