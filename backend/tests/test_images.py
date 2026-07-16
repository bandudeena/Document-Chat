from app.rag.images import DESCRIBE_PROMPT, ImageDescriber, filter_images, inspect_image
from app.rag.loader import PageElement
from tests.conftest import FakeChat, make_png


def img_element(page=1, **png_kwargs) -> PageElement:
    return PageElement(kind="image", page=page, image_bytes=make_png(**png_kwargs))


def test_inspect_image_reads_dimensions_and_mime():
    info = inspect_image(make_png(width=300, height=150))
    assert info == (300, 150, "image/png")


def test_inspect_image_rejects_garbage():
    assert inspect_image(b"not an image at all") is None


def test_filter_drops_small_images():
    elements = [
        img_element(width=200, height=200),  # kept
        img_element(width=20, height=20),  # too small
    ]
    kept, skipped = filter_images(elements, min_px=64, min_bytes=10, max_count=10)
    assert len(kept) == 1
    assert skipped == 1
    assert kept[0][1] == "image/png"


def test_filter_respects_max_count():
    elements = [img_element() for _ in range(5)]
    kept, skipped = filter_images(elements, min_px=8, min_bytes=10, max_count=2)
    assert len(kept) == 2
    assert skipped == 3


def test_filter_ignores_non_image_elements():
    elements = [PageElement(kind="text", page=1, text="hi"), img_element()]
    kept, _ = filter_images(elements, min_px=8, min_bytes=10, max_count=10)
    assert len(kept) == 1


def test_describer_returns_llm_text():
    describer = ImageDescriber(FakeChat(response="A red rectangle logo."))
    assert describer.describe(make_png(), "image/png") == "A red rectangle logo."


def test_describe_prompt_mentions_charts_and_text():
    # the prompt is part of the behaviour contract: it must ask for embedded text/data
    assert "text" in DESCRIBE_PROMPT.lower()
    assert "chart" in DESCRIBE_PROMPT.lower()
