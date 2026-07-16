import pytest

from app.rag.loader import (
    EmptyDocumentError,
    PageElement,
    UnsupportedFileTypeError,
    load_elements,
)


def test_txt_and_md_load_as_single_text_element():
    for name in ("notes.txt", "notes.md"):
        elements = load_elements(b"Hello world.\nSecond line.", name)
        assert len(elements) == 1
        assert elements[0].kind == "text"
        assert "Second line." in elements[0].text


def test_unsupported_extension_raises():
    with pytest.raises(UnsupportedFileTypeError):
        load_elements(b"binary", "slides.pptx")


def test_empty_text_file_raises():
    with pytest.raises(EmptyDocumentError):
        load_elements(b"   \n  ", "empty.txt")


def test_pdf_text_and_table_extraction(table_pdf):
    elements = load_elements(table_pdf, "report.pdf")
    kinds = {el.kind for el in elements}
    assert "text" in kinds
    assert "table" in kinds

    table = next(el for el in elements if el.kind == "table")
    assert table.page == 1
    assert "| Region | Revenue | Growth |" in table.text
    assert "| EMEA | 120 | 4% |" in table.text


def test_pdf_image_extraction(image_pdf):
    elements = load_elements(image_pdf, "diagram.pdf")
    images = [el for el in elements if el.kind == "image"]
    assert len(images) == 1
    assert images[0].page == 1
    assert len(images[0].image_bytes) > 100
    # prose on the page is still captured
    assert any("ingestion flow" in el.text for el in elements if el.kind == "text")


def test_page_element_defaults():
    el = PageElement(kind="text", page=2, text="x")
    assert el.image_bytes == b""
