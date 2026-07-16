"""Structured logging and request tracing.

Every log line is a single JSON object so the output can be shipped to
CloudWatch / Loki without extra parsing. A request id is attached to every
line emitted while handling a request (propagated via a contextvar, returned
to the client in the ``X-Request-ID`` header).

The middleware is pure ASGI (not ``BaseHTTPMiddleware``) on purpose: the
contextvar must stay set while a streaming (SSE) response body is being
generated, and BaseHTTPMiddleware runs the downstream app in a separate task.
"""

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

logger = logging.getLogger("rag")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        fields = getattr(record, "event_fields", None)
        if fields:
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # uvicorn's access log is redundant with our per-request event
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def log_event(name: str, **fields) -> None:
    """Emit a structured event, e.g. ``log_event("chat_completed", tokens=123)``."""
    logger.info(name, extra={"event_fields": fields})


class RequestContextMiddleware:
    """Assigns a request id, times the request, and logs one line per API request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = Headers(scope=scope).get("x-request-id") or uuid.uuid4().hex[:12]
        request_id_var.set(request_id)
        started = time.perf_counter()
        path: str = scope["path"]
        status_code: int | None = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                if path.startswith("/api"):
                    log_event(
                        "request",
                        method=scope["method"],
                        path=path,
                        status=status_code,
                        duration_ms=round((time.perf_counter() - started) * 1000, 1),
                    )
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            log_event(
                "request_failed",
                method=scope["method"],
                path=path,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            raise
