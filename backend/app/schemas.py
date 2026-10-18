from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)  # bcrypt ignores bytes past 72


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    email: str


class UploadResponse(BaseModel):
    doc_id: str
    chunks: int


class DocumentsResponse(BaseModel):
    doc_ids: list[str]


class HistoryTurn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    doc_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    history: list[HistoryTurn] = Field(default_factory=list)


class CitationOut(BaseModel):
    source_doc: str
    page_number: int
    relevant_quote: str
    verified: bool
