# Stage 1 — build the React frontend
FROM node:20-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2 — Python runtime serving API + built frontend
FROM python:3.12-slim AS runtime
WORKDIR /app

# faiss needs libgomp at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project > requirements.txt \
    && uv pip install --system -r requirements.txt \
    && rm requirements.txt

COPY backend/app ./app
COPY --from=frontend /build/dist ./static

RUN useradd --create-home appuser \
    && mkdir -p /app/data && chown -R appuser:appuser /app/data
USER appuser

ENV RAG_FRONTEND_DIST=/app/static \
    RAG_DATA_DIR=/app/data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/healthz')"

CMD ["uvicorn", "--factory", "app.main:create_app", "--host", "0.0.0.0", "--port", "8000"]
