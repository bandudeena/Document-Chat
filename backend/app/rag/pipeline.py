"""The RAG pipeline: ingestion and streaming question answering.

Both AWS-touching collaborators (embeddings via the store, chat LLM, image
describer) are injected, so the pipeline is fully testable offline.

`answer()` is an async generator of typed events, mirrored 1:1 as SSE events
by the API layer:
  {"type": "sources", "sources": [...]}      -- retrieved citations, sent first
  {"type": "token",   "text": "..."}         -- streamed answer fragments
  {"type": "done",    "usage": {...}}        -- token counts + cost estimate
"""

import hashlib
import logging
import time
import uuid
from collections.abc import AsyncIterator

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

from app.config import Settings
from app.observability import log_event
from app.rag.chunker import chunk_elements
from app.rag.images import ImageDescriber, filter_images
from app.rag.loader import load_elements
from app.rag.prompts import (
    AGENT_SYSTEM_PROMPT,
    NOT_FOUND_ANSWER,
    build_messages,
    format_tool_result,
    trim_history,
)
from app.rag.store import VectorStore
from app.schemas import ChatTurn, DocumentInfo, IngestStats, Source

logger = logging.getLogger("rag.pipeline")

_SNIPPET_CHARS = 400


class DuplicateDocumentError(ValueError):
    """Raised when the exact same file content has already been ingested."""

    def __init__(self, existing: DocumentInfo):
        self.existing = existing
        super().__init__(
            f"This file is already in the library as '{existing.filename}' "
            f"(uploaded {existing.created_at})"
        )


class search_documents(BaseModel):  # noqa: N801 - lowercase class name IS the tool name
    """Search the user's uploaded documents for relevant content. Returns the most
    relevant chunks (text passages, tables, image descriptions) with citation numbers."""

    query: str = Field(
        description="A focused, self-contained search query. Resolve any conversational "
        "references so the query works on its own."
    )


class RagPipeline:
    def __init__(
        self,
        store: VectorStore,
        llm: BaseChatModel,
        describer: ImageDescriber,
        settings: Settings,
    ):
        self.store = store
        self._llm = llm
        self._describer = describer
        self._settings = settings

    # -- ingestion -------------------------------------------------------------

    def ingest(self, data: bytes, filename: str) -> DocumentInfo:
        started = time.perf_counter()

        # Duplicate check by content hash — cheap, and runs BEFORE any parsing
        # or paid vision/embedding calls.
        content_sha256 = hashlib.sha256(data).hexdigest()
        existing = self.store.find_by_hash(content_sha256)
        if existing is not None:
            log_event("duplicate_upload_rejected", filename=filename, existing=existing.filename)
            raise DuplicateDocumentError(existing)

        doc_id = uuid.uuid4().hex[:12]
        elements = load_elements(data, filename)

        images, skipped = filter_images(
            elements,
            min_px=self._settings.image_min_px,
            min_bytes=self._settings.image_min_bytes,
            max_count=self._settings.image_max_per_doc,
        )
        described = 0
        for element, mime_type in images:
            try:
                description = self._describer.describe(element.image_bytes, mime_type)
            except Exception:
                logger.warning("image description failed (page %s)", element.page, exc_info=True)
                skipped += 1
                continue
            if description:
                element.text = f"Image on page {element.page}: {description}"
                described += 1
            else:
                skipped += 1

        documents = chunk_elements(
            elements,
            doc_id=doc_id,
            filename=filename,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        pages = max((el.page or 1 for el in elements), default=0)
        stats = IngestStats(
            pages=pages,
            chunks=len(documents),
            text_chunks=sum(1 for d in documents if d.metadata["content_type"] == "text"),
            tables=sum(1 for d in documents if d.metadata["content_type"] == "table"),
            images_described=described,
            images_skipped=skipped,
        )
        info = self.store.add_document(
            doc_id, filename, documents, stats, content_sha256=content_sha256
        )
        log_event(
            "document_ingested",
            doc_id=doc_id,
            filename=filename,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            **stats.model_dump(),
        )
        return info

    # -- question answering -------------------------------------------------------

    async def answer(self, question: str, history: list[ChatTurn]) -> AsyncIterator[dict]:
        if self._settings.retrieval_mode == "agentic":
            async for event in self._answer_agentic(question, history):
                yield event
        else:
            async for event in self._answer_simple(question, history):
                yield event

    async def _answer_agentic(self, question: str, history: list[ChatTurn]) -> AsyncIterator[dict]:
        """LLM-driven retrieval: the model formulates queries and searches via a
        tool, possibly multiple times, then answers from what it found.

        Event order differs from simple mode: `searching` events stream as the
        agent works, `sources` arrives at the end (we only then know everything
        it retrieved). Bounded by agent_max_rounds/agent_max_searches.
        """
        started = time.perf_counter()
        settings = self._settings

        messages: list[BaseMessage] = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
        for turn in trim_history(history, settings.history_max_turns):
            cls = HumanMessage if turn.role == "user" else AIMessage
            messages.append(cls(content=turn.content))
        messages.append(HumanMessage(content=question))

        # The tool stays bound on every round: Bedrock Converse requires the tool
        # config to be present whenever the history contains tool-use blocks.
        # Termination is steered through tool results ("answer now") instead.
        llm_with_tool = self._llm.bind_tools([search_documents])
        retrieved: dict[str, tuple[int, Document, float]] = {}  # chunk key -> (ref, doc, score)
        searches = 0
        input_tokens = output_tokens = 0
        streamed_any_text = False

        for round_no in range(settings.agent_max_rounds):
            full: AIMessageChunk | None = None
            async for chunk in llm_with_tool.astream(messages):
                text = _chunk_text(chunk)
                if text:
                    streamed_any_text = True
                    yield {"type": "token", "text": text}
                if isinstance(chunk, AIMessageChunk):
                    if chunk.usage_metadata:
                        input_tokens += chunk.usage_metadata.get("input_tokens", 0)
                        output_tokens += chunk.usage_metadata.get("output_tokens", 0)
                    full = chunk if full is None else full + chunk

            tool_calls = (full.tool_calls or []) if full is not None else []
            last_round = round_no == settings.agent_max_rounds - 1
            if not tool_calls or last_round:
                break

            messages.append(full)
            next_is_final = round_no == settings.agent_max_rounds - 2
            for call in tool_calls:
                query = str(call.get("args", {}).get("query", "")).strip()
                yield {"type": "searching", "query": query}
                if searches >= settings.agent_max_searches:
                    result = "Search limit reached."
                else:
                    hits = self.store.search(
                        query, k=settings.top_k, score_threshold=settings.score_threshold
                    )
                    new_refs: list[tuple[int, Document]] = []
                    for doc, score in hits:
                        key = f"{doc.metadata['doc_id']}:{doc.metadata['chunk_index']}"
                        if key not in retrieved:
                            retrieved[key] = (len(retrieved) + 1, doc, score)
                        new_refs.append((retrieved[key][0], doc))
                    result = format_tool_result(new_refs)
                searches += 1
                if next_is_final or searches >= settings.agent_max_searches:
                    result += (
                        "\n\n(No further searches are available — answer the question now "
                        "using the results you already have.)"
                    )
                messages.append(ToolMessage(content=result, tool_call_id=call.get("id") or ""))

        if not streamed_any_text:
            # Every round ended in tool calls (shouldn't happen with the steering
            # above, but never leave the user with an empty answer).
            yield {"type": "token", "text": NOT_FOUND_ANSWER}

        sources = [
            Source(
                ref=ref,
                doc_id=doc.metadata["doc_id"],
                filename=doc.metadata["filename"],
                page=doc.metadata.get("page"),
                content_type=doc.metadata.get("content_type", "text"),
                snippet=doc.page_content[:_SNIPPET_CHARS],
                score=round(score, 4),
            )
            for ref, doc, score in sorted(retrieved.values(), key=lambda item: item[0])
        ]
        yield {"type": "sources", "sources": [s.model_dump() for s in sources]}

        usage = self._usage(input_tokens, output_tokens)
        yield {"type": "done", "usage": usage}
        log_event(
            "chat_completed",
            mode="agentic",
            searches=searches,
            retrieved=len(retrieved),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            **usage,
        )

    async def _answer_simple(self, question: str, history: list[ChatTurn]) -> AsyncIterator[dict]:
        started = time.perf_counter()
        hits = self.store.search(
            question,
            k=self._settings.top_k,
            score_threshold=self._settings.score_threshold,
        )

        if not hits:
            # Guardrail: nothing relevant retrieved -> deterministic answer,
            # no LLM call, no chance to hallucinate.
            yield {"type": "sources", "sources": []}
            yield {"type": "token", "text": NOT_FOUND_ANSWER}
            yield {"type": "done", "usage": self._usage(0, 0)}
            log_event("chat_no_context", question_chars=len(question))
            return

        sources = [
            Source(
                ref=ref,
                doc_id=doc.metadata["doc_id"],
                filename=doc.metadata["filename"],
                page=doc.metadata.get("page"),
                content_type=doc.metadata.get("content_type", "text"),
                snippet=doc.page_content[:_SNIPPET_CHARS],
                score=round(score, 4),
            )
            for ref, (doc, score) in enumerate(hits, start=1)
        ]
        yield {"type": "sources", "sources": [s.model_dump() for s in sources]}

        messages = build_messages(
            question,
            history,
            [doc for doc, _ in hits],
            history_max_turns=self._settings.history_max_turns,
        )

        input_tokens = 0
        output_tokens = 0
        async for chunk in self._llm.astream(messages):
            text = _chunk_text(chunk)
            if text:
                yield {"type": "token", "text": text}
            if isinstance(chunk, AIMessageChunk) and chunk.usage_metadata:
                input_tokens += chunk.usage_metadata.get("input_tokens", 0)
                output_tokens += chunk.usage_metadata.get("output_tokens", 0)

        usage = self._usage(input_tokens, output_tokens)
        yield {"type": "done", "usage": usage}
        log_event(
            "chat_completed",
            mode="simple",
            retrieved=len(hits),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            **usage,
        )

    def _usage(self, input_tokens: int, output_tokens: int) -> dict:
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": self._settings.estimate_cost_usd(input_tokens, output_tokens),
        }


def _chunk_text(chunk) -> str:
    """Extract plain text from a streamed chunk (content may be str or block list)."""
    content = chunk.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)
