"""Split extracted elements into embeddable chunks.

Strategy:
- Prose is split recursively (paragraph -> sentence -> word boundaries) at
  ~1500 characters (~375 tokens) with 200 overlap. Big enough that a chunk
  carries a self-contained idea, small enough that top-5 retrieval fits
  comfortably in the prompt.
- Tables and image descriptions are kept ATOMIC (never split): half a table
  is worse than no table, and descriptions are already short.
- Every chunk gets a CONTEXTUAL HEADER (source file, document title, page)
  prepended before embedding. Splitting strips a chunk of its document-level
  identity ("Prague store declined 3%" no longer mentions the company or the
  report); the header puts those tokens back so queries naming the document's
  subject match again. Zero-LLM-cost variant of Anthropic's contextual
  retrieval (they report -35% retrieval failures for the full version).

Every chunk carries the metadata needed for citations:
{doc_id, filename, page, content_type, chunk_index}.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.loader import PageElement

_TITLE_MAX_CHARS = 120


def extract_title(elements: list[PageElement]) -> str:
    """Best-effort document title: first line of the first text element."""
    for element in elements:
        if element.kind == "text" and element.text.strip():
            first_line = element.text.strip().splitlines()[0].strip().lstrip("# ")
            return first_line[:_TITLE_MAX_CHARS]
    return ""


def contextual_header(filename: str, title: str, page: int | None) -> str:
    source = filename
    if title and title.lower() not in filename.lower():
        source += f' — "{title}"'
    if page is not None:
        source += f", page {page}"
    return f"[Source: {source}]"


def chunk_elements(
    elements: list[PageElement],
    *,
    doc_id: str,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    title = extract_title(elements)
    documents: list[Document] = []
    for element in elements:
        if not element.text.strip():
            continue
        metadata = {
            "doc_id": doc_id,
            "filename": filename,
            "page": element.page,
            "content_type": element.kind,
        }
        header = contextual_header(filename, title, element.page)
        # text is split recursively; table markdown / image descriptions stay atomic
        is_prose = element.kind == "text"
        pieces = splitter.split_text(element.text) if is_prose else [element.text]
        documents.extend(
            Document(page_content=f"{header}\n{piece}", metadata=dict(metadata))
            for piece in pieces
        )

    for index, doc in enumerate(documents):
        doc.metadata["chunk_index"] = index
    return documents
