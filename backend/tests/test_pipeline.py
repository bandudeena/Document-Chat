import pytest
from langchain_core.messages import AIMessage

from app.config import Settings
from app.rag.images import ImageDescriber
from app.rag.pipeline import DuplicateDocumentError, RagPipeline
from app.rag.prompts import NOT_FOUND_ANSWER
from app.rag.store import VectorStore
from tests.conftest import FakeAgenticChat, FakeChat


async def collect(pipeline, question, history=None):
    return [event async for event in pipeline.answer(question, history or [])]


def test_ingest_pdf_with_table_and_prose(pipeline, table_pdf):
    info = pipeline.ingest(table_pdf, "report.pdf")
    assert info.stats.tables == 1
    assert info.stats.text_chunks >= 1
    assert info.stats.chunks == info.stats.text_chunks + info.stats.tables
    assert pipeline.store.document_count == 1


def test_ingest_pdf_with_image_describes_it(pipeline, image_pdf):
    info = pipeline.ingest(image_pdf, "diagram.pdf")
    assert info.stats.images_described == 1
    # the description (FakeChat's canned answer) is now retrievable
    hits = pipeline.store.search("Fake grounded answer", k=3, score_threshold=0.05)
    assert any(d.metadata["content_type"] == "image" for d, _ in hits)


def test_ingest_survives_describer_failure(pipeline, image_pdf, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bedrock down")

    monkeypatch.setattr(pipeline._describer, "describe", boom)
    info = pipeline.ingest(image_pdf, "diagram.pdf")  # must not raise
    assert info.stats.images_described == 0
    assert info.stats.images_skipped >= 1


def test_ingest_rejects_duplicate_content(pipeline):
    pipeline.ingest(b"The warranty period is 24 months.", "warranty.txt")

    # same bytes, even under a different filename -> duplicate
    with pytest.raises(DuplicateDocumentError) as exc_info:
        pipeline.ingest(b"The warranty period is 24 months.", "renamed-copy.txt")
    assert "warranty.txt" in str(exc_info.value)
    assert pipeline.store.document_count == 1

    # different content under the same filename is a new document
    pipeline.ingest(b"The warranty period is 36 months.", "warranty.txt")
    assert pipeline.store.document_count == 2


def test_ingest_after_delete_allows_reupload(pipeline):
    info = pipeline.ingest(b"reupload me", "a.txt")
    pipeline.store.delete_document(info.doc_id)
    pipeline.ingest(b"reupload me", "a.txt")  # must not raise
    assert pipeline.store.document_count == 1


async def test_answer_with_no_relevant_context_short_circuits(pipeline):
    events = await collect(pipeline, "question about nothing ingested")
    types = [e["type"] for e in events]
    assert types == ["sources", "token", "done"]
    assert events[0]["sources"] == []
    assert events[1]["text"] == NOT_FOUND_ANSWER
    assert events[2]["usage"]["input_tokens"] == 0  # no LLM call happened


async def test_answer_streams_tokens_sources_and_usage(pipeline):
    pipeline.ingest(b"The warranty period is 24 months from purchase.", "warranty.txt")

    events = await collect(pipeline, "how long is the warranty period")
    assert events[0]["type"] == "sources"
    sources = events[0]["sources"]
    assert sources and sources[0]["filename"] == "warranty.txt"
    assert sources[0]["ref"] == 1
    assert 0 < sources[0]["score"] <= 1.001

    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert "".join(tokens) == "Fake grounded answer citing [1]."

    done = events[-1]
    assert done["type"] == "done"
    assert done["usage"]["input_tokens"] == 120
    assert done["usage"]["output_tokens"] == 15
    assert done["usage"]["estimated_cost_usd"] == pytest.approx(
        120 * 1.0 / 1e6 + 15 * 5.0 / 1e6
    )


async def test_answer_respects_top_k(pipeline, settings):
    for i in range(10):
        pipeline.ingest(f"warranty clause number {i} lasts months".encode(), f"doc{i}.txt")
    events = await collect(pipeline, "warranty clause months")
    assert len(events[0]["sources"]) <= settings.top_k


# --- agentic mode ---------------------------------------------------------------


def tool_call(query, call_id="t1"):
    return {"name": "search_documents", "args": {"query": query}, "id": call_id}


def make_agentic_pipeline(fake_embeddings, tmp_path, script, **settings_overrides):
    settings = Settings(
        **{
            "data_dir": tmp_path / "agentic-data",
            "retrieval_mode": "agentic",
            "score_threshold": 0.05,
            **settings_overrides,
        }
    )
    store = VectorStore(fake_embeddings, settings.data_dir)
    llm = FakeAgenticChat(script=script)
    return RagPipeline(store, llm, ImageDescriber(FakeChat()), settings)


async def test_agentic_search_then_answer(fake_embeddings, tmp_path):
    script = [
        AIMessage(content="", tool_calls=[tool_call("warranty period duration")]),
        AIMessage(content="The warranty period is 24 months. [1]"),
    ]
    pipeline = make_agentic_pipeline(fake_embeddings, tmp_path, script)
    pipeline.ingest(b"The warranty period is 24 months from purchase.", "warranty.txt")

    events = await collect(pipeline, "how long is the warranty?")
    types = [e["type"] for e in events]

    # agent searched with its own reformulated query
    searching = next(e for e in events if e["type"] == "searching")
    assert searching["query"] == "warranty period duration"

    # sources arrive at the end, after the streamed answer
    assert types[-2:] == ["sources", "done"]
    sources = events[types.index("sources")]["sources"]
    assert sources and sources[0]["filename"] == "warranty.txt"
    assert sources[0]["ref"] == 1

    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert answer == "The warranty period is 24 months. [1]"

    # usage accumulated over both LLM rounds
    done = events[-1]
    assert done["usage"]["input_tokens"] == 200
    assert done["usage"]["output_tokens"] == 20


async def test_agentic_multi_search_dedupes_refs(fake_embeddings, tmp_path):
    script = [
        AIMessage(content="", tool_calls=[tool_call("alpha topic", "t1")]),
        AIMessage(content="", tool_calls=[tool_call("alpha topic details", "t2")]),
        AIMessage(content="Combined answer. [1]"),
    ]
    pipeline = make_agentic_pipeline(fake_embeddings, tmp_path, script)
    pipeline.ingest(b"alpha topic details are described here", "alpha.txt")

    events = await collect(pipeline, "tell me about alpha")
    searches = [e for e in events if e["type"] == "searching"]
    assert len(searches) == 2

    # both searches hit the same chunk -> ONE source with a stable ref
    sources = next(e for e in events if e["type"] == "sources")["sources"]
    assert len(sources) == 1
    assert sources[0]["ref"] == 1


async def test_agentic_no_results_leads_to_not_found_style_answer(fake_embeddings, tmp_path):
    script = [
        AIMessage(content="", tool_calls=[tool_call("bitcoin price")]),
        AIMessage(content="I couldn't find anything about that in your uploaded documents."),
    ]
    pipeline = make_agentic_pipeline(
        fake_embeddings, tmp_path, script, score_threshold=0.99
    )
    pipeline.ingest(b"completely unrelated content", "other.txt")

    events = await collect(pipeline, "what is the bitcoin price?")
    sources = next(e for e in events if e["type"] == "sources")["sources"]
    assert sources == []
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert "couldn't find" in answer


async def test_agentic_respects_search_cap(fake_embeddings, tmp_path):
    script = [
        AIMessage(
            content="",
            tool_calls=[tool_call("q1", "t1"), tool_call("q2", "t2"), tool_call("q3", "t3")],
        ),
        AIMessage(content="Answer with what I have. [1]"),
    ]
    pipeline = make_agentic_pipeline(
        fake_embeddings, tmp_path, script, agent_max_searches=2
    )
    pipeline.ingest(b"q1 q2 q3 relevant content", "doc.txt")

    events = await collect(pipeline, "question")
    # all three tool calls acknowledged, but only 2 real searches allowed
    assert len([e for e in events if e["type"] == "searching"]) == 3
    assert events[-1]["type"] == "done"  # loop terminated cleanly


async def test_agentic_round_cap_forces_answer(fake_embeddings, tmp_path):
    # script keeps asking to search; the last configured round must answer
    # (pipeline drops the tool on the final round; the fake then emits its last message)
    script = [
        AIMessage(content="", tool_calls=[tool_call("q1")]),
        AIMessage(content="", tool_calls=[tool_call("q2")]),
        AIMessage(content="Forced final answer."),
    ]
    pipeline = make_agentic_pipeline(
        fake_embeddings, tmp_path, script, agent_max_rounds=3
    )
    pipeline.ingest(b"some content", "doc.txt")

    events = await collect(pipeline, "question")
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert answer == "Forced final answer."
    assert events[-1]["type"] == "done"
