"""Email + password auth. Sessions are opaque random tokens hashed into a
`login_sessions` table (not JWT) — the same pattern already proven in
week-06-profilefit: trivial to revoke on logout (just delete the row), no
signing secret to manage, no token-expiry-without-revocation gap.
"""

import hashlib
import secrets
import time

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.db import session
from app.models import LoginSession, User
from app.schemas import LoginRequest, SignupRequest, UserOut

router = APIRouter(prefix="/api/auth", tags=["authentication"])
COOKIE = "docuchat_session"
SESSION_LIFETIME_SECONDS = 30 * 86400


def digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _issue_session(response: Response, db: Session, user: User) -> None:
    token = secrets.token_urlsafe(48)
    db.add(LoginSession(token_hash=digest(token), user_id=user.id, expires_at=int(time.time()) + SESSION_LIFETIME_SECONDS))
    db.execute(delete(LoginSession).where(LoginSession.expires_at < int(time.time())))
    db.commit()
    response.set_cookie(
        COOKIE,
        token,
        httponly=True,
        secure=settings().secure,
        samesite="lax",
        max_age=SESSION_LIFETIME_SECONDS,
        path="/",
    )


def current_user(request: Request, db: Session = Depends(session)) -> User:
    token = request.cookies.get(COOKIE)
    login = db.get(LoginSession, digest(token)) if token and len(token) < 200 else None
    if not login or login.expires_at <= time.time():
        raise HTTPException(401, "Sign in to continue.")
    user = db.get(User, login.user_id)
    if not user:
        raise HTTPException(401, "Your session has expired. Please sign in again.")
    return user


@router.post("/signup", response_model=UserOut)
def signup(body: SignupRequest, response: Response, db: Session = Depends(session)):
    hashed = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
    user = User(email=body.email.lower(), hashed_password=hashed)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "An account with that email already exists.")
    _issue_session(response, db, user)
    return UserOut(id=user.id, email=user.email)


@router.post("/login", response_model=UserOut)
def login(body: LoginRequest, response: Response, db: Session = Depends(session)):
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if not user or not bcrypt.checkpw(body.password.encode(), user.hashed_password.encode()):
        raise HTTPException(401, "Incorrect email or password.")
    _issue_session(response, db, user)
    return UserOut(id=user.id, email=user.email)


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(session)):
    token = request.cookies.get(COOKIE)
    if token:
        db.execute(delete(LoginSession).where(LoginSession.token_hash == digest(token)))
        db.commit()
    response.delete_cookie(COOKIE, path="/")
    return {"status": "ok"}


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return UserOut(id=user.id, email=user.email)
