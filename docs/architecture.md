# Architecture

One container, two pipelines.

```mermaid
flowchart TB
    subgraph browser [Browser — React SPA]
        UI[Chat UI + Library sidebar]
    end

    subgraph container [FastAPI container]
        API[REST + SSE API]
        SPA[Static SPA via app.frontend]

        subgraph ingest [Ingestion pipeline]
            LOAD[loader: pdfplumber text + tables, pypdf images]
            TBL[tables → Markdown]
            IMG[images → filter → Haiku vision description]
            CHUNK[chunker: recursive 1500/200, tables & images atomic]
        end

        subgraph query [Query pipeline]
            RET[retrieve top-k, cosine ≥ threshold]
            PROMPT[grounded prompt + citations + history]
        end

        STORE[(FAISS index + registry.json\npersisted volume)]
    end

    subgraph aws [AWS Bedrock]
        TITAN[Titan Embed v2]
        HAIKU[Claude Haiku 4.5\ntext + vision]
    end

    UI -->|upload| API --> LOAD --> TBL --> CHUNK
    LOAD --> IMG --> CHUNK
    IMG -->|describe| HAIKU
    CHUNK -->|embed| TITAN
    CHUNK --> STORE

    UI -->|question| API --> RET --> PROMPT -->|stream| HAIKU
    RET -->|embed query| TITAN
    RET --> STORE
    HAIKU -->|SSE tokens + sources + usage| UI
```

## Request flows

**Ingestion** (`POST /api/documents`): file → per-page extraction (prose via
pdfplumber, tables converted to Markdown, embedded images via pypdf) → small
images filtered out → remaining images described by Haiku vision → everything
chunked (prose split recursively at ~1500 chars / 200 overlap; tables and image
descriptions kept atomic) → embedded with Titan v2 → stored in FAISS with
`{doc_id, filename, page, content_type, chunk_index}` metadata → index +
registry persisted to the Docker volume.

**Query** (`POST /api/chat`, SSE): question embedded → top-5 cosine search with
a relevance floor (0.15, calibrated empirically) → if nothing passes, a
deterministic "not found" answer is returned **without an LLM call** → else the
chunks become numbered context blocks in a grounded system prompt (plus the
last 6 trimmed history turns) → Haiku streams the answer → the client receives
`sources` (citations with scores), `token` fragments, and a final `done` event
with token counts and estimated cost.

## Why one container

FastAPI serves the built React app itself (`app.frontend("/", directory=...)`,
API routes take priority, unknown paths fall back to `index.html`). No nginx,
no CORS, one port, one image — the right amount of infrastructure for this
scope. The production section of the README describes how this splits apart
when it needs to scale.
