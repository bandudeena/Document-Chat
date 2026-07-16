"""API request/response models."""

from typing import Literal

from pydantic import BaseModel, Field


class IngestStats(BaseModel):
    pages: int = 0
    chunks: int = 0
    text_chunks: int = 0
    tables: int = 0
    images_described: int = 0
    images_skipped: int = 0


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    created_at: str
    stats: IngestStats


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[ChatTurn] = Field(default_factory=list)


class Source(BaseModel):
    ref: int  # the [n] number used in the answer
    doc_id: str
    filename: str
    page: int | None
    content_type: Literal["text", "table", "image"]
    snippet: str
    score: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    documents: int
    chunks: int
