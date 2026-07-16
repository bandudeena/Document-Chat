"""App factory.

Run with:  uvicorn --factory app.main:create_app
Tests call create_app(settings=..., pipeline=<fake>) to run fully offline.

The built React app (frontend/dist) is served by FastAPI itself via
``app.frontend()`` — API routes take priority, unknown paths fall back to
index.html for client-side routing. One process, one container, one port.
"""

from pathlib import Path

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings
from app.observability import RequestContextMiddleware, configure_logging, log_event
from app.rag.pipeline import RagPipeline


def build_pipeline(settings: Settings) -> RagPipeline:
    """Wire the real AWS-backed collaborators. Imported lazily so tests never touch boto3."""
    from langchain_aws import BedrockEmbeddings, ChatBedrockConverse

    from app.rag.images import ImageDescriber
    from app.rag.store import VectorStore

    embeddings = BedrockEmbeddings(
        model_id=settings.embedding_model_id, region_name=settings.aws_region
    )
    llm = ChatBedrockConverse(
        model_id=settings.chat_model_id, region_name=settings.aws_region, max_tokens=2048
    )
    store = VectorStore(embeddings, settings.data_dir)
    return RagPipeline(store, llm, ImageDescriber(llm), settings)


def create_app(settings: Settings | None = None, pipeline: RagPipeline | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging()

    app = FastAPI(title="Chat With Your Docs", version="0.1.0")
    app.add_middleware(RequestContextMiddleware)

    app.state.settings = settings
    app.state.pipeline = pipeline or build_pipeline(settings)

    app.include_router(router)

    dist = settings.frontend_dist or Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if dist.is_dir():
        app.frontend("/", directory=dist)
    else:
        log_event("frontend_not_built", dist=str(dist))

    return app
