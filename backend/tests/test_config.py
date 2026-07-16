from app.config import Settings


def test_defaults_are_sane():
    s = Settings()
    assert s.chunk_overlap < s.chunk_size
    assert 0 < s.score_threshold < 1
    assert s.top_k > 0
    assert s.max_upload_bytes == s.max_upload_mb * 1024 * 1024


def test_env_override(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "9")
    monkeypatch.setenv("RAG_CHAT_MODEL_ID", "anthropic.claude-haiku-4-5-20251001-v1:0")
    s = Settings()
    assert s.top_k == 9
    assert s.chat_model_id.startswith("anthropic.")


def test_region_falls_back_to_aws_env(monkeypatch):
    monkeypatch.delenv("RAG_AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    assert Settings().aws_region == "eu-west-1"


def test_cost_estimate():
    s = Settings(price_input_per_mtok=1.0, price_output_per_mtok=5.0)
    # 1M input + 1M output = $6
    assert s.estimate_cost_usd(1_000_000, 1_000_000) == 6.0
    assert s.estimate_cost_usd(0, 0) == 0.0
