from app.rag.chunker import chunk_elements, contextual_header, extract_title
from app.rag.loader import PageElement


def make_chunks(elements, chunk_size=200, chunk_overlap=50):
    return chunk_elements(
        elements, doc_id="d1", filename="f.pdf", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def body(doc) -> str:
    """Chunk text without its contextual header (first line)."""
    return doc.page_content.split("\n", 1)[1]


def test_long_text_is_split_with_overlap():
    text = " ".join(f"word{i}" for i in range(200))  # ~1400 chars
    docs = make_chunks([PageElement(kind="text", page=1, text=text)])
    assert len(docs) > 1
    assert all(len(body(d)) <= 200 for d in docs)
    # overlap: consecutive chunks share content
    assert body(docs[0]).split()[-1] in body(docs[1])


def test_every_chunk_gets_a_contextual_header():
    text = " ".join(f"word{i}" for i in range(200))
    docs = make_chunks(
        [
            PageElement(kind="text", page=1, text="Annual Report 2025\nintro"),
            PageElement(kind="text", page=2, text=text),
        ]
    )
    for doc in docs:
        header = doc.page_content.split("\n", 1)[0]
        assert header.startswith("[Source: f.pdf")
        assert '"Annual Report 2025"' in header  # doc title recovered into every chunk
    assert ", page 2]" in docs[-1].page_content.split("\n", 1)[0]


def test_table_is_never_split():
    rows = "\n".join(f"| r{i} | v{i} |" for i in range(100))
    table_md = "| A | B |\n| --- | --- |\n" + rows  # far beyond chunk_size
    docs = make_chunks([PageElement(kind="table", page=2, text=table_md)])
    assert len(docs) == 1
    assert docs[0].page_content.endswith(table_md)  # atomic, header prepended


def test_image_description_is_atomic():
    docs = make_chunks([PageElement(kind="image", page=3, text="Image on page 3: a chart.")])
    assert len(docs) == 1
    assert docs[0].metadata["content_type"] == "image"


def test_extract_title():
    elements = [
        PageElement(kind="table", page=1, text="| A |\n| --- |"),
        PageElement(kind="text", page=1, text="# My Great Title\nbody text"),
    ]
    assert extract_title(elements) == "My Great Title"  # markdown heading stripped
    assert extract_title([]) == ""
    long_line = "x" * 300
    assert len(extract_title([PageElement(kind="text", page=1, text=long_line)])) == 120


def test_contextual_header_variants():
    assert contextual_header("f.pdf", "Title", 3) == '[Source: f.pdf — "Title", page 3]'
    assert contextual_header("f.pdf", "", None) == "[Source: f.pdf]"
    # title redundant with filename -> not repeated
    assert contextual_header("annual-report.pdf", "Annual-Report", 1) == (
        "[Source: annual-report.pdf, page 1]"
    )


def test_metadata_and_chunk_index():
    docs = make_chunks(
        [
            PageElement(kind="text", page=1, text="short text"),
            PageElement(kind="table", page=2, text="| A |\n| --- |\n| 1 |"),
        ]
    )
    assert [d.metadata["chunk_index"] for d in docs] == list(range(len(docs)))
    assert docs[0].metadata == {
        "doc_id": "d1",
        "filename": "f.pdf",
        "page": 1,
        "content_type": "text",
        "chunk_index": 0,
    }


def test_empty_and_whitespace_elements_are_dropped():
    docs = make_chunks([PageElement(kind="text", page=1, text="   \n ")])
    assert docs == []
