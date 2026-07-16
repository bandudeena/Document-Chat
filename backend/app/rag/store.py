"""Hybrid store: FAISS vector search + BM25 keyword search, fused with RRF.

Dense embeddings are weak exactly where keyword search is strong (part
numbers, "clause 4.2", rare exact terms) and vice versa, so `search()` runs
both and merges the rankings with Reciprocal Rank Fusion — the standard
hybrid-retrieval pattern (RAGFlow's "multiple recall + fused re-ranking",
Anthropic reports ~-49%% retrieval failures for hybrid vs dense-only).

FAISS uses inner product over L2-normalised vectors (= cosine); BM25 is
in-process over the same chunks and rebuilt on writes (corpora here are small;
production would swap this class for OpenSearch/pgvector, which do hybrid
natively). The registry maps doc_id -> {filename, chunk_ids, stats, ...} so we
can list documents and delete all chunks belonging to one document.
"""

import json
import logging
import re
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

with warnings.catch_warnings():
    # langchain-community is sunset, but it is still the canonical home of the
    # FAISS integration (the standalone langchain-faiss package is an empty
    # placeholder as of writing). Revisit when the integration actually moves.
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.schemas import DocumentInfo, IngestStats

logger = logging.getLogger("rag.store")

_INDEX_DIR = "faiss_index"
_REGISTRY_FILE = "registry.json"
_RRF_K = 60  # standard RRF damping constant
_CANDIDATE_MULTIPLIER = 4  # each retriever proposes k * this candidates before fusion


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{2,}", text.lower())


def rrf_fuse(rankings: list[list[str]], k: int = _RRF_K) -> list[str]:
    """Reciprocal Rank Fusion: merge ranked id lists; ids ranked high anywhere win.

    score(id) = sum over rankings of 1 / (k + rank). Pure function for testability.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)


class VectorStore:
    def __init__(self, embeddings: Embeddings, data_dir: Path):
        self._embeddings = embeddings
        self._data_dir = data_dir
        self._index_path = data_dir / _INDEX_DIR
        self._registry_path = data_dir / _REGISTRY_FILE
        self._faiss: FAISS | None = None
        self._registry: dict[str, dict] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._load()
        self._rebuild_bm25()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        if self._registry_path.exists():
            self._registry = json.loads(self._registry_path.read_text())
        if (self._index_path / "index.faiss").exists():
            # Our own artifact from save_local below — safe to deserialize.
            self._faiss = FAISS.load_local(
                str(self._index_path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("loaded FAISS index with %d chunks", self.chunk_count)

    def _persist(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        if self._faiss is not None:
            self._faiss.save_local(str(self._index_path))
        self._registry_path.write_text(json.dumps(self._registry, indent=2))

    # -- write path ------------------------------------------------------------

    def find_by_hash(self, content_sha256: str) -> DocumentInfo | None:
        """Return the already-ingested document with identical content, if any."""
        for doc_id, record in self._registry.items():
            if record.get("content_sha256") == content_sha256:
                return self._to_info(doc_id, record)
        return None

    def add_document(
        self,
        doc_id: str,
        filename: str,
        documents: list[Document],
        stats: IngestStats,
        content_sha256: str = "",
    ) -> DocumentInfo:
        if not documents:
            raise ValueError("cannot add a document with zero chunks")
        ids = [f"{doc_id}:{i}" for i in range(len(documents))]

        if self._faiss is None:
            with warnings.catch_warnings():
                # langchain warns that normalize_L2 is "not applicable" to
                # MAX_INNER_PRODUCT, but normalisation IS applied to stored and
                # query vectors — which is exactly what makes IP == cosine.
                warnings.simplefilter("ignore", UserWarning)
                self._faiss = FAISS.from_documents(
                    documents,
                    self._embeddings,
                    ids=ids,
                    normalize_L2=True,
                    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                )
        else:
            self._faiss.add_documents(documents, ids=ids)

        record = {
            "filename": filename,
            "chunk_ids": ids,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "stats": stats.model_dump(),
            "content_sha256": content_sha256,
        }
        self._registry[doc_id] = record
        self._rebuild_bm25()
        self._persist()
        return self._to_info(doc_id, record)

    def delete_document(self, doc_id: str) -> bool:
        record = self._registry.pop(doc_id, None)
        if record is None:
            return False
        if self._faiss is not None:
            self._faiss.delete(record["chunk_ids"])
        self._rebuild_bm25()
        self._persist()
        return True

    def _rebuild_bm25(self) -> None:
        """BM25 index over all chunks. Rebuilt on writes — corpora here are small
        enough that a full rebuild (milliseconds) beats incremental bookkeeping."""
        if self._faiss is None or self.chunk_count == 0:
            self._bm25, self._bm25_ids = None, []
            return
        ids, corpus = [], []
        for chunk_id, doc in self._faiss.docstore._dict.items():
            ids.append(chunk_id)
            corpus.append(_tokenize(doc.page_content))
        self._bm25_ids = ids
        self._bm25 = BM25Okapi(corpus)

    # -- read path ---------------------------------------------------------------

    def search(self, query: str, *, k: int, score_threshold: float) -> list[tuple[Document, float]]:
        """Hybrid top-k: vector + BM25 candidates fused with RRF.

        Returned score is always the cosine similarity (so it stays comparable
        and thresholdable). Gate: a candidate passes with cosine >= threshold;
        a keyword (BM25) candidate gets a relaxed bar (0.6x threshold) — exact-
        term matches deserve benefit of the doubt, but pure token noise
        ("the"/"of" overlap on an off-topic query) still gets filtered, which
        preserves the not-found guardrail.
        """
        if self._faiss is None or self.chunk_count == 0:
            return []
        n_candidates = max(k * _CANDIDATE_MULTIPLIER, k)

        # -- vector candidates (raw faiss: gives us ids + cosine directly)
        query_vec = np.array([self._embeddings.embed_query(query)], dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm
        sims, positions = self._faiss.index.search(query_vec, min(n_candidates, self.chunk_count))
        position_of = {cid: pos for pos, cid in self._faiss.index_to_docstore_id.items()}
        vector_scores: dict[str, float] = {}
        vector_ranked: list[str] = []
        for pos, sim in zip(positions[0], sims[0], strict=True):
            if pos == -1:
                continue
            chunk_id = self._faiss.index_to_docstore_id[int(pos)]
            vector_scores[chunk_id] = float(sim)
            vector_ranked.append(chunk_id)

        # -- keyword candidates
        bm25_ranked: list[str] = []
        if self._bm25 is not None:
            bm25_scores = self._bm25.get_scores(_tokenize(query))
            order = np.argsort(bm25_scores)[::-1][:n_candidates]
            bm25_ranked = [self._bm25_ids[i] for i in order if bm25_scores[i] > 0]
        bm25_set = set(bm25_ranked)

        # -- fuse, gate, assemble
        results: list[tuple[Document, float]] = []
        for chunk_id in rrf_fuse([vector_ranked, bm25_ranked]):
            sim = vector_scores.get(chunk_id)
            if sim is None:  # BM25-only candidate: compute its true cosine
                stored = self._faiss.index.reconstruct(position_of[chunk_id])
                sim = float(np.dot(query_vec[0], stored))
            bar = score_threshold * 0.6 if chunk_id in bm25_set else score_threshold
            if sim < bar:
                continue
            doc = self._faiss.docstore.search(chunk_id)
            if isinstance(doc, Document):
                results.append((doc, sim))
            if len(results) == k:
                break
        return results

    def list_documents(self) -> list[DocumentInfo]:
        return [self._to_info(doc_id, rec) for doc_id, rec in sorted(self._registry.items())]

    @property
    def document_count(self) -> int:
        return len(self._registry)

    @property
    def chunk_count(self) -> int:
        if self._faiss is None:
            return 0
        return self._faiss.index.ntotal

    @staticmethod
    def _to_info(doc_id: str, record: dict) -> DocumentInfo:
        return DocumentInfo(
            doc_id=doc_id,
            filename=record["filename"],
            created_at=record["created_at"],
            stats=IngestStats(**record["stats"]),
        )
