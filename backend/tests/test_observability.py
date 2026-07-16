import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.observability import (
    JsonFormatter,
    RequestContextMiddleware,
    log_event,
    request_id_var,
)


@pytest.fixture
def capture(monkeypatch):
    """Route the 'rag' logger through JsonFormatter into a list of parsed dicts."""
    records: list[dict] = []

    class Sink(logging.Handler):
        def emit(self, record):
            records.append(json.loads(JsonFormatter().format(record)))

    logger = logging.getLogger("rag")
    sink = Sink()
    logger.addHandler(sink)
    logger.setLevel(logging.INFO)
    yield records
    logger.removeHandler(sink)


def test_log_event_emits_structured_fields(capture):
    log_event("chat_completed", input_tokens=10, estimated_cost_usd=0.0001)
    assert capture[-1]["message"] == "chat_completed"
    assert capture[-1]["input_tokens"] == 10
    assert capture[-1]["estimated_cost_usd"] == 0.0001
    assert "ts" in capture[-1] and "level" in capture[-1]


def test_request_id_flows_into_logs_and_response(capture):
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/api/ping")
    def ping():
        log_event("inside_handler")
        return {"ok": True}

    client = TestClient(app)
    response = client.get("/api/ping", headers={"X-Request-ID": "abc123"})

    assert response.headers["x-request-id"] == "abc123"
    handler_line = next(r for r in capture if r["message"] == "inside_handler")
    request_line = next(r for r in capture if r["message"] == "request")
    assert handler_line["request_id"] == "abc123"
    assert request_line["request_id"] == "abc123"
    assert request_line["status"] == 200
    assert request_line["duration_ms"] >= 0


def test_request_id_generated_when_absent():
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/api/ping")
    def ping():
        return {"rid": request_id_var.get()}

    client = TestClient(app)
    response = client.get("/api/ping")
    rid = response.headers["x-request-id"]
    assert rid != "-" and len(rid) == 12
    assert response.json()["rid"] == rid


def test_non_api_paths_not_logged(capture):
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/assets/x.js")
    def asset():
        return {}

    TestClient(app).get("/assets/x.js")
    assert not any(r["message"] == "request" for r in capture)
