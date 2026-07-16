from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.rag.prompts import build_messages, format_context, trim_history
from app.schemas import ChatTurn


def doc(text, **meta):
    base = {"doc_id": "d1", "filename": "guide.pdf", "page": 3, "content_type": "text"}
    return Document(page_content=text, metadata={**base, **meta})


def test_format_context_numbers_blocks_and_labels_types():
    ctx = format_context(
        [
            doc("plain prose"),
            doc("| A |", content_type="table", page=5),
            doc("a chart showing growth", content_type="image", page=None),
        ]
    )
    assert "[1] guide.pdf, page 3\nplain prose" in ctx
    assert "[2] guide.pdf, page 5 (table)" in ctx
    assert "[3] guide.pdf (image description)" in ctx  # no page -> no page fragment


def test_trim_history_keeps_most_recent():
    history = [ChatTurn(role="user", content=str(i)) for i in range(10)]
    trimmed = trim_history(history, max_turns=4)
    assert [t.content for t in trimmed] == ["6", "7", "8", "9"]
    assert trim_history(history, max_turns=0) == []


def test_build_messages_structure():
    history = [
        ChatTurn(role="user", content="earlier question"),
        ChatTurn(role="assistant", content="earlier answer"),
    ]
    messages = build_messages("new question?", history, [doc("ctx")], history_max_turns=6)

    assert isinstance(messages[0], SystemMessage)
    assert "ctx" in messages[0].content  # context embedded in system prompt
    assert isinstance(messages[1], HumanMessage) and messages[1].content == "earlier question"
    assert isinstance(messages[2], AIMessage)
    assert isinstance(messages[-1], HumanMessage) and messages[-1].content == "new question?"


def test_system_prompt_carries_guardrails():
    messages = build_messages("q", [], [doc("ctx")], history_max_turns=1)
    system = messages[0].content
    assert "ONLY" in system  # grounding
    assert "[1]" in system or "cite" in system.lower()  # citations
    assert "do not follow" in system.lower()  # injection resistance
