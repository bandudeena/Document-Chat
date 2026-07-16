"""Application settings.

Everything tunable lives here so the RAG behaviour (chunk sizes, retrieval
thresholds, model ids) can be changed without touching code. Values come from
environment variables prefixed with ``RAG_`` (or a local ``.env`` file).
"""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # --- AWS / models -----------------------------------------------------
    aws_region: str = Field(
        default="us-east-1",
        validation_alias=AliasChoices("RAG_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"),
    )
    # Cross-region inference profile id; fall back to the bare model id if your
    # account has direct access instead (see scripts/smoke_bedrock.py).
    chat_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"

    # --- storage ----------------------------------------------------------
    data_dir: Path = Path("data")
    frontend_dist: Path | None = None

    # --- chunking / retrieval ---------------------------------------------
    # "agentic": the LLM drives retrieval via a search tool — it formulates its
    #   own queries (resolving conversational context) and can search multiple
    #   times for multi-hop questions. Better recall, more LLM calls.
    # "simple": one embedding search on the raw question, then answer. Cheaper,
    #   fully deterministic retrieval, off-topic questions never hit the LLM.
    retrieval_mode: Literal["agentic", "simple"] = "agentic"
    agent_max_searches: int = 4  # total search-tool calls per question
    agent_max_rounds: int = 3  # LLM round-trips; the last round must answer

    chunk_size: int = 1500  # characters (~375 tokens)
    chunk_overlap: int = 200
    top_k: int = 5
    # Cosine similarity floor; hits below this are considered irrelevant and
    # we answer "not found" without calling the LLM. Calibrated empirically
    # against Titan v2: on-topic chunks score ~0.20-0.45, off-topic ~<0.08.
    score_threshold: float = 0.15

    # --- request limits / guardrails ---------------------------------------
    max_upload_mb: int = 20
    max_question_chars: int = 2000
    history_max_turns: int = 6

    # --- image handling -----------------------------------------------------
    image_min_px: int = 64  # skip decorative images smaller than this on either side
    image_min_bytes: int = 2048
    image_max_per_doc: int = 20  # cost cap on vision calls per document

    # --- pricing (USD per 1M tokens) for cost logging ------------------------
    price_input_per_mtok: float = 1.00
    price_output_per_mtok: float = 5.00

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def estimate_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return round(
            input_tokens * self.price_input_per_mtok / 1_000_000
            + output_tokens * self.price_output_per_mtok / 1_000_000,
            6,
        )
