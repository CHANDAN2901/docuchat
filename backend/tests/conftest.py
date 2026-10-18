import os
import tempfile
import uuid

os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mktemp(suffix='.db')}"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import main as main_module  # noqa: E402
from app.chain import new_store  # noqa: E402
from app.db import Base, engine  # noqa: E402

Base.metadata.create_all(engine())
_test_store = new_store(client_options={"location": ":memory:"})
main_module.get_store = lambda: _test_store


@pytest.fixture
def client():
    return TestClient(main_module.app)


@pytest.fixture
def signed_up_client(client):
    email = f"{uuid.uuid4()}@example.com"  # unique per test — the SQLite test DB persists across the whole session
    resp = client.post("/api/auth/signup", json={"email": email, "password": "correct-horse"})
    assert resp.status_code == 200
    return client
