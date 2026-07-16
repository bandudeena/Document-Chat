"""Extract structured elements (text, tables, images) from uploaded files.

The loader is deliberately format-aware rather than using a generic
"one string per document" loader: tables and images need different treatment
downstream (tables stay atomic as Markdown, images get described by a vision
model), so we surface them as distinct elements with page metadata for
citations.
"""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pdfplumber
from pypdf import PdfReader

from app.rag.tables import table_to_markdown

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

ElementKind = Literal["text", "table", "image"]


class UnsupportedFileTypeError(ValueError):
    def __init__(self, extension: str):
        self.extension = extension
        super().__init__(
            f"Unsupported file type '{extension}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )


class EmptyDocumentError(ValueError):
    pass


@dataclass
class PageElement:
    kind: ElementKind
    page: int | None
    text: str = ""  # extracted text / table Markdown / image description (filled later)
    image_bytes: bytes = field(default=b"", repr=False)


def load_elements(data: bytes, filename: str) -> list[PageElement]:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(ext)

    if ext in {".txt", ".md"}:
        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            raise EmptyDocumentError(f"'{filename}' contains no text")
        return [PageElement(kind="text", page=None, text=text)]

    return _load_pdf(data, filename)


def _load_pdf(data: bytes, filename: str) -> list[PageElement]:
    elements: list[PageElement] = []

    # Text + tables via pdfplumber (best table detection of the
    # permissively-licensed parsers).
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                elements.append(PageElement(kind="text", page=page_no, text=text))
            for table in page.extract_tables():
                markdown = table_to_markdown(table)
                if markdown:
                    elements.append(PageElement(kind="table", page=page_no, text=markdown))

    # Embedded images via pypdf (pdfplumber does not expose decoded image bytes).
    # Extraction failures on exotic encodings are skipped, not fatal.
    reader = PdfReader(io.BytesIO(data))
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            images = page.images
        except Exception:
            continue
        for img in images:
            try:
                raw = img.data
            except Exception:
                continue
            if raw:
                elements.append(PageElement(kind="image", page=page_no, image_bytes=raw))

    if not elements:
        raise EmptyDocumentError(
            f"No extractable content in '{filename}' (scanned/image-only PDFs need OCR, "
            "which is out of scope)"
        )
    return elements
