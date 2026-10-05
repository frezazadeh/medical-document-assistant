from typing import Literal

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: str
    filename: str
    pages: int
    chunks: int
    uploaded_at: str


class UploadOut(DocumentOut):
    created: bool = Field(description="False if the same file had been uploaded before")


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[str] | None = Field(
        default=None, description="Restrict the search to these documents. Default: all."
    )
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)


class Citation(BaseModel):
    ref: int = Field(description="The number used in the answer text, e.g. [2]")
    document_id: str
    filename: str
    page: int
    snippet: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool = Field(
        description=(
            "True if every citation resolves to a retrieved source and the numbers in each "
            "cited sentence appear in that source (or if nothing was found)"
        )
    )
    unsupported: list[str] = Field(
        default_factory=list,
        description="Sentences with a number that is in none of the passages they cite",
    )
    trace_id: str
    model: str
    latency_ms: int


class HealthOut(BaseModel):
    status: str
    llm_provider: str
    llm_model: str
    embedding_model: str
    documents: int
