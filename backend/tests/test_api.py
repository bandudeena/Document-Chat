import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client(settings, pipeline):
    app = create_app(settings=settings, pipeline=pipeline)
    return TestClient(app)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in lines:
            events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_healthz(client):
    response = client.get("/api/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "documents": 0, "chunks": 0}


def test_upload_list_delete_roundtrip(client):
    upload = client.post(
        "/api/documents",
        files={"file": ("facts.txt", b"The Eiffel Tower is 330 metres tall.", "text/plain")},
    )
    assert upload.status_code == 201
    doc = upload.json()
    assert doc["filename"] == "facts.txt"
    assert doc["stats"]["chunks"] == 1

    listed = client.get("/api/documents").json()
    assert [d["doc_id"] for d in listed] == [doc["doc_id"]]

    assert client.delete(f"/api/documents/{doc['doc_id']}").status_code == 204
    assert client.get("/api/documents").json() == []


def test_upload_pdf_with_table(client, table_pdf):
    response = client.post(
        "/api/documents", files={"file": ("report.pdf", table_pdf, "application/pdf")}
    )
    assert response.status_code == 201
    assert response.json()["stats"]["tables"] == 1


def test_upload_rejections(client, settings):
    # unsupported type
    r = client.post("/api/documents", files={"file": ("x.docx", b"data", "application/msword")})
    assert r.status_code == 415

    # empty file
    r = client.post("/api/documents", files={"file": ("x.txt", b"", "text/plain")})
    assert r.status_code == 400

    # oversize
    big = b"a" * (settings.max_upload_bytes + 1)
    r = client.post("/api/documents", files={"file": ("big.txt", big, "text/plain")})
    assert r.status_code == 413

    # whitespace-only content
    r = client.post("/api/documents", files={"file": ("blank.txt", b"  \n ", "text/plain")})
    assert r.status_code == 422


def test_upload_duplicate_returns_409(client):
    payload = {"file": ("facts.txt", b"The Eiffel Tower is 330 metres tall.", "text/plain")}
    assert client.post("/api/documents", files=payload).status_code == 201

    duplicate = client.post(
        "/api/documents",
        files={"file": ("copy.txt", b"The Eiffel Tower is 330 metres tall.", "text/plain")},
    )
    assert duplicate.status_code == 409
    assert "already in the library" in duplicate.json()["detail"]
    assert len(client.get("/api/documents").json()) == 1


def test_delete_unknown_document(client):
    assert client.delete("/api/documents/nope").status_code == 404


def test_chat_streams_sse_events(client):
    client.post(
        "/api/documents",
        files={"file": ("warranty.txt", b"The warranty period is 24 months.", "text/plain")},
    )
    with client.stream(
        "POST", "/api/chat", json={"question": "how long is the warranty period"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = parse_sse("".join(response.iter_text()))

    names = [name for name, _ in events]
    assert names[0] == "sources"
    assert names[-1] == "done"
    assert "token" in names

    sources = events[0][1]["sources"]
    assert sources[0]["filename"] == "warranty.txt"
    answer = "".join(data["text"] for name, data in events if name == "token")
    assert answer == "Fake grounded answer citing [1]."
    assert events[-1][1]["usage"]["output_tokens"] == 15


def test_chat_without_documents_says_not_found(client):
    with client.stream("POST", "/api/chat", json={"question": "anything at all"}) as response:
        events = parse_sse("".join(response.iter_text()))
    assert events[0][1]["sources"] == []
    answer = "".join(d["text"] for n, d in events if n == "token")
    assert "couldn't find" in answer


def test_chat_validation(client, settings):
    assert client.post("/api/chat", json={"question": "   "}).status_code == 400
    too_long = "x" * (settings.max_question_chars + 1)
    assert client.post("/api/chat", json={"question": too_long}).status_code == 413
    # malformed history role
    r = client.post(
        "/api/chat",
        json={"question": "q", "history": [{"role": "system", "content": "hack"}]},
    )
    assert r.status_code == 422
