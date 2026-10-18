def test_signup_then_me_returns_the_new_user(client):
    resp = client.post("/api/auth/signup", json={"email": "new@example.com", "password": "correct-horse"})
    assert resp.status_code == 200
    assert resp.json()["email"] == "new@example.com"

    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == "new@example.com"


def test_signup_rejects_duplicate_email(client):
    client.post("/api/auth/signup", json={"email": "dup@example.com", "password": "correct-horse"})

    resp = client.post("/api/auth/signup", json={"email": "dup@example.com", "password": "another-pass"})

    assert resp.status_code == 409


def test_login_rejects_wrong_password(client):
    client.post("/api/auth/signup", json={"email": "bob@example.com", "password": "correct-horse"})
    client.post("/api/auth/logout")

    resp = client.post("/api/auth/login", json={"email": "bob@example.com", "password": "wrong-password"})

    assert resp.status_code == 401


def test_login_accepts_correct_password_after_logout(client):
    client.post("/api/auth/signup", json={"email": "carol@example.com", "password": "correct-horse"})
    client.post("/api/auth/logout")

    resp = client.post("/api/auth/login", json={"email": "carol@example.com", "password": "correct-horse"})

    assert resp.status_code == 200
    assert client.get("/api/auth/me").status_code == 200


def test_protected_routes_401_without_a_session(client):
    assert client.get("/api/documents").status_code == 401
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 401


def test_logout_invalidates_the_session(client):
    client.post("/api/auth/signup", json={"email": "dave@example.com", "password": "correct-horse"})
    assert client.get("/api/auth/me").status_code == 200

    client.post("/api/auth/logout")

    assert client.get("/api/auth/me").status_code == 401
