"""HTTP API.

POST   /api/documents        multipart upload -> ingest -> DocumentInfo
GET    /api/documents        list ingested documents
DELETE /api/documents/{id}   remove a document and its chunks
POST   /api/chat             SSE stream: sources -> token* -> done
GET    /api/healthz          liveness + corpus stats

SSE is hand-rolled (``event: X\\ndata: json\\n\\n`` over a StreamingResponse):
it is ~10 lines and keeps the endpoint fully testable with TestClient.
"""

import json
import logging

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.rag.loader import EmptyDocumentError, UnsupportedFileTypeError
from app.rag.pipeline import DuplicateDocumentError, RagPipeline
from app.schemas import ChatRequest, DocumentInfo, HealthResponse

logger = logging.getLogger("rag.api")

router = APIRouter(prefix="/api")


def _pipeline(request: Request) -> RagPipeline:
    return request.app.state.pipeline


@router.get("/healthz", response_model=HealthResponse)
def healthz(request: Request) -> HealthResponse:
    pipeline = _pipeline(request)
    return HealthResponse(
        status="ok",
        documents=pipeline.store.document_count,
        chunks=pipeline.store.chunk_count,
    )


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(request: Request) -> list[DocumentInfo]:
    return _pipeline(request).store.list_documents()


@router.post("/documents", response_model=DocumentInfo, status_code=201)
async def upload_document(request: Request, file: UploadFile) -> DocumentInfo:
    settings = request.app.state.settings
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, f"File exceeds the {settings.max_upload_mb} MB limit")
    if not data:
        raise HTTPException(400, "File is empty")

    pipeline = _pipeline(request)
    try:
        # Ingestion is blocking (PDF parsing + embedding calls); keep the event loop free.
        return await run_in_threadpool(pipeline.ingest, data, file.filename or "upload")
    except UnsupportedFileTypeError as exc:
        raise HTTPException(415, str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(422, str(exc)) from exc
    except DuplicateDocumentError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(request: Request, doc_id: str) -> None:
    if not _pipeline(request).store.delete_document(doc_id):
        raise HTTPException(404, f"Unknown document '{doc_id}'")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    settings = request.app.state.settings
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty")
    if len(question) > settings.max_question_chars:
        raise HTTPException(413, f"Question exceeds {settings.max_question_chars} characters")

    pipeline = _pipeline(request)

    async def stream():
        try:
            async for event in pipeline.answer(question, body.history):
                yield _sse(event["type"], event)
        except Exception:
            logger.exception("chat stream failed")
            yield _sse("error", {"type": "error", "message": "Something went wrong answering."})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
