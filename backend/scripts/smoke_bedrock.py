"""Live smoke test against AWS Bedrock — run once before first use.

Verifies the three calls the app makes:
  1. Titan v2 embedding
  2. Claude Haiku streaming completion (resolves inference-profile vs bare model id)
  3. Claude Haiku vision (image description)

Usage:  python scripts/smoke_bedrock.py
"""

import base64
import io
import sys

from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_core.messages import HumanMessage
from PIL import Image

sys.path.insert(0, ".")
from app.config import Settings  # noqa: E402


def main() -> int:
    settings = Settings()
    print(f"region={settings.aws_region}")

    # 1. embeddings
    embeddings = BedrockEmbeddings(
        model_id=settings.embedding_model_id, region_name=settings.aws_region
    )
    vector = embeddings.embed_query("hello bedrock")
    print(f"[1/3] embeddings ok: {settings.embedding_model_id} dim={len(vector)}")

    # 2. chat completion — try configured id, then the bare fallback
    candidates = [settings.chat_model_id]
    if settings.chat_model_id.startswith("us."):
        candidates.append(settings.chat_model_id.removeprefix("us."))
    llm = None
    for model_id in candidates:
        try:
            llm = ChatBedrockConverse(
                model_id=model_id, region_name=settings.aws_region, max_tokens=64
            )
            chunks = [c for c in llm.stream("Reply with exactly: pong")]
            text = "".join(getattr(c, "text", "") or "" for c in chunks)
            print(f"[2/3] chat ok: {model_id} -> {text.strip()!r}")
            if model_id != settings.chat_model_id:
                print(f"      NOTE: set RAG_CHAT_MODEL_ID={model_id}")
            break
        except Exception as exc:
            print(f"      {model_id} failed: {exc}")
            llm = None
    if llm is None:
        print("[2/3] chat FAILED for all candidate model ids")
        return 1

    # 3. vision
    buf = io.BytesIO()
    Image.new("RGB", (120, 80), (200, 60, 40)).save(buf, format="PNG")
    message = HumanMessage(
        content=[
            {
                "type": "image",
                "source_type": "base64",
                "data": base64.b64encode(buf.getvalue()).decode(),
                "mime_type": "image/png",
            },
            {"type": "text", "text": "What colour is this image? One word."},
        ]
    )
    response = llm.invoke([message])
    print(f"[3/3] vision ok: {str(response.text).strip()!r}")
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
