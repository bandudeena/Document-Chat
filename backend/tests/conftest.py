"""Shared test fakes and fixtures.

Nothing in the test suite talks to AWS:
- FakeEmbeddings: deterministic bag-of-words hashing — texts sharing words get
  similar vectors, so retrieval behaves realistically.
- FakeChat: a minimal LangChain chat model with configurable response and
  usage metadata; streams word by word like the real thing.
- Fixture PDFs are generated on the fly with reportlab (one with a real ruled
  table, one with an embedded image), so no binaries live in the repo.
"""

import io
import json
import math
from collections.abc import Iterator

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from PIL import Image
from pydantic import PrivateAttr

from app.config import Settings
from app.rag.images import ImageDescriber
from app.rag.pipeline import RagPipeline
from app.rag.store import VectorStore

DIM = 64


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embedding: shared tokens => higher cosine sim."""

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for token in text.lower().split():
            vec[hash(token) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeChat(BaseChatModel):
    """Streams `response` word by word; attaches usage metadata to the last chunk."""

    response: str = "Fake grounded answer citing [1]."
    input_tokens: int = 120
    output_tokens: int = 15

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _usage(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = AIMessage(content=self.response, usage_metadata=self._usage())
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        words = self.response.split(" ")
        for i, word in enumerate(words):
            text = word if i == 0 else " " + word
            usage = self._usage() if i == len(words) - 1 else None
            yield ChatGenerationChunk(message=AIMessageChunk(content=text, usage_metadata=usage))


class FakeAgenticChat(BaseChatModel):
    """Plays back a script of AIMessages (which may carry tool_calls), one per
    round, streaming text word by word — mimics an agent that searches then answers."""

    script: list[AIMessage]
    input_tokens: int = 100
    output_tokens: int = 10
    _cursor: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "fake-agentic-chat"

    def bind_tools(self, tools, **kwargs):
        return self  # tool schema is irrelevant to the scripted playback

    def _next(self) -> AIMessage:
        message = self.script[min(self._cursor, len(self.script) - 1)]
        self._cursor += 1
        return message

    def _usage(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    def _stream(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        message = self._next()
        words = message.content.split(" ") if message.content else []
        for i, word in enumerate(words):
            text = word if i == 0 else " " + word
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))
        tool_call_chunks = [
            {
                "name": call["name"],
                "args": json.dumps(call["args"]),
                "id": call["id"],
                "index": i,
                "type": "tool_call_chunk",
            }
            for i, call in enumerate(message.tool_calls)
        ]
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="", tool_call_chunks=tool_call_chunks, usage_metadata=self._usage()
            )
        )


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def fake_chat() -> FakeChat:
    return FakeChat()


@pytest.fixture
def settings(tmp_path) -> Settings:
    # simple mode here: most tests assert the deterministic single-search
    # behaviour; agentic mode has its own fixtures in test_pipeline.
    return Settings(
        data_dir=tmp_path / "data",
        retrieval_mode="simple",
        score_threshold=0.05,
        image_min_bytes=100,
    )


@pytest.fixture
def store(fake_embeddings, settings) -> VectorStore:
    return VectorStore(fake_embeddings, settings.data_dir)


@pytest.fixture
def pipeline(store, fake_chat, settings) -> RagPipeline:
    return RagPipeline(store, fake_chat, ImageDescriber(fake_chat), settings)


# --- fixture files ------------------------------------------------------------


def make_png(width: int = 200, height: int = 120, color=(200, 60, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def table_pdf() -> bytes:
    """PDF with a paragraph and a ruled 3x3 table (pdfplumber detects ruled tables)."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    table = Table(
        [
            ["Region", "Revenue", "Growth"],
            ["EMEA", "120", "4%"],
            ["APAC", "95", "9%"],
        ]
    )
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.8, colors.black)]))
    doc.build(
        [
            Paragraph("Quarterly results overview for the sales organisation.", styles["Normal"]),
            Spacer(1, 12),
            table,
        ]
    )
    return buf.getvalue()


@pytest.fixture(scope="session")
def image_pdf() -> bytes:
    """PDF with a paragraph and one embedded PNG image."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica", 11)
    c.drawString(72, 780, "The architecture diagram below shows the ingestion flow.")
    c.drawImage(ImageReader(io.BytesIO(make_png())), 72, 600, width=200, height=120)
    c.save()
    return buf.getvalue()
