"""Filter extracted PDF images and describe them with a vision LLM.

Images can't be embedded with a text embedding model, so we make their content
searchable by generating a textual description with Claude Haiku (vision) at
ingestion time. The description is then chunked/embedded like any other text.

The describer takes any LangChain chat model, so tests inject a fake and no
AWS call happens offline.
"""

import base64
import io
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from PIL import Image

from app.rag.loader import PageElement

logger = logging.getLogger("rag.images")

DESCRIBE_PROMPT = (
    "You are indexing a document for search. Describe this image factually and "
    "completely: what it shows, any text it contains, and any data points, labels "
    "or trends if it is a chart or diagram. Do not speculate beyond what is visible. "
    "Answer with the description only."
)

_PIL_TO_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "WEBP": "image/webp"}


def inspect_image(data: bytes) -> tuple[int, int, str] | None:
    """Return (width, height, mime_type) or None if not a decodable image."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            mime = _PIL_TO_MIME.get(im.format or "")
            if mime is None:
                return None
            return im.width, im.height, mime
    except Exception:
        return None


def filter_images(
    elements: list[PageElement], *, min_px: int, min_bytes: int, max_count: int
) -> tuple[list[tuple[PageElement, str]], int]:
    """Drop decorative/undecodable images.

    Returns (kept as (element, mime_type) pairs, skipped_count). Small images
    are almost always logos, bullets or separators — describing them wastes
    vision-model calls and pollutes retrieval.
    """
    kept: list[tuple[PageElement, str]] = []
    skipped = 0
    for el in elements:
        if el.kind != "image":
            continue
        info = inspect_image(el.image_bytes)
        if (
            info is None
            or len(el.image_bytes) < min_bytes
            or info[0] < min_px
            or info[1] < min_px
            or len(kept) >= max_count
        ):
            skipped += 1
            continue
        kept.append((el, info[2]))
    return kept, skipped


class ImageDescriber:
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def describe(self, image_bytes: bytes, mime_type: str) -> str:
        message = HumanMessage(
            content=[
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": base64.b64encode(image_bytes).decode(),
                    "mime_type": mime_type,
                },
                {"type": "text", "text": DESCRIBE_PROMPT},
            ]
        )
        response = self._llm.invoke([message])
        return str(response.text).strip()
