from langchain_core.documents import Document

from app.rag.store import VectorStore, rrf_fuse
from app.schemas import IngestStats


def doc(text, **meta):
    base = {"doc_id": "d1", "filename": "f.txt", "page": 1, "content_type": "text"}
    return Document(page_content=text, metadata={**base, **meta})


def add(store, doc_id, texts, filename="f.txt"):
    docs = [doc(t, doc_id=doc_id, filename=filename) for t in texts]
    return store.add_document(doc_id, filename, docs, IngestStats(chunks=len(docs)))


def test_add_and_search_returns_relevant_chunk_first(store):
    add(store, "d1", ["the quarterly revenue grew in EMEA", "penguins live in antarctica"])
    results = store.search("what was the EMEA revenue", k=2, score_threshold=0.05)
    assert results
    top_doc, score = results[0]
    assert "revenue" in top_doc.page_content
    assert 0 < score <= 1.001


def test_threshold_filters_unrelated_queries(store):
    add(store, "d1", ["penguins live in antarctica"])
    assert store.search("zzz qqq xxx unrelated", k=3, score_threshold=0.9) == []


def test_search_on_empty_store(store):
    assert store.search("anything", k=5, score_threshold=0.0) == []
    assert store.chunk_count == 0
    assert store.document_count == 0


def test_delete_document_removes_its_chunks(store):
    add(store, "d1", ["alpha content"])
    add(store, "d2", ["beta content"])
    assert store.chunk_count == 2

    assert store.delete_document("d1") is True
    assert store.chunk_count == 1
    assert [d.doc_id for d in store.list_documents()] == ["d2"]
    # deleted content no longer retrievable
    hits = store.search("alpha content", k=5, score_threshold=0.05)
    assert all(d.metadata["doc_id"] != "d1" for d, _ in hits)


def test_delete_unknown_document_returns_false(store):
    assert store.delete_document("nope") is False


def test_persistence_roundtrip(fake_embeddings, settings):
    store1 = VectorStore(fake_embeddings, settings.data_dir)
    add(store1, "d1", ["persistent knowledge about gravity"])

    store2 = VectorStore(fake_embeddings, settings.data_dir)  # fresh load from disk
    assert store2.document_count == 1
    assert store2.chunk_count == 1
    results = store2.search("knowledge about gravity", k=1, score_threshold=0.05)
    assert results and "gravity" in results[0][0].page_content


def test_find_by_hash(store):
    docs = [doc("some content", doc_id="d1")]
    store.add_document("d1", "f.txt", docs, IngestStats(chunks=1), content_sha256="abc123")

    found = store.find_by_hash("abc123")
    assert found is not None and found.doc_id == "d1"
    assert store.find_by_hash("other") is None

    # deleting the document frees the hash for re-upload
    store.delete_document("d1")
    assert store.find_by_hash("abc123") is None


def test_find_by_hash_tolerates_legacy_records(store):
    # records written before hashing existed have no content_sha256 key
    add(store, "old", ["legacy content"])
    store._registry["old"].pop("content_sha256", None)
    assert store.find_by_hash("anything") is None


def test_rrf_fuse_rewards_presence_in_both_rankings():
    fused = rrf_fuse([["a", "b", "c"], ["c", "d"]])
    # c is mid/top in both lists -> beats b (one list only)
    assert fused.index("c") < fused.index("b")
    # a is rank 1 in the first list -> still strong
    assert fused[0] in {"a", "c"}
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_fuse_handles_empty_rankings():
    assert rrf_fuse([[], []]) == []
    assert rrf_fuse([["x"], []]) == ["x"]


def test_hybrid_search_finds_exact_rare_terms(store):
    add(store, "d1", ["the replacement part number is XJ-9000 for the compressor"])
    add(store, "d2", ["penguins live in antarctica and eat fish"])
    results = store.search("XJ-9000", k=2, score_threshold=0.05)
    assert results
    assert "XJ-9000" in results[0][0].page_content


def test_hybrid_gate_still_refuses_common_token_noise(store):
    # off-topic query sharing only stopword-ish tokens must not pass the gate
    add(store, "d1", ["the warranty for the device covers parts and labour"])
    assert store.search("what is the capital of France", k=3, score_threshold=0.5) == []


def test_list_documents_exposes_stats(store):
    add(store, "d1", ["one", "two"])
    infos = store.list_documents()
    assert len(infos) == 1
    assert infos[0].stats.chunks == 2
    assert infos[0].created_at  # ISO timestamp recorded
