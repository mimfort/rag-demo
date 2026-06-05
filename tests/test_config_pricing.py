import pytest


def _base_env(monkeypatch):
    # Минимум обязательных переменных, чтобы load_settings() не падал.
    monkeypatch.setenv("LLM_API_KEY", "sk-or-test")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")


def test_pricing_fields_parsed(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("LLM_MAX_PROMPT_PRICE_PER_MTOK", "3.0")
    monkeypatch.setenv("LLM_MAX_COMPLETION_PRICE_PER_MTOK", "15")
    monkeypatch.setenv("LLM_PROVIDER_SORT", "price")

    import rag.config as config
    s = config.load_settings()
    assert s.llm_fallback_model == "openai/gpt-4o-mini"
    assert s.llm_max_prompt_price == 3.0
    assert s.llm_max_completion_price == 15.0
    assert s.llm_provider_sort == "price"


def test_pricing_fields_default_empty(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.delenv("LLM_FALLBACK_MODEL", raising=False)
    monkeypatch.delenv("LLM_MAX_PROMPT_PRICE_PER_MTOK", raising=False)
    monkeypatch.delenv("LLM_MAX_COMPLETION_PRICE_PER_MTOK", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_SORT", raising=False)

    import rag.config as config
    s = config.load_settings()
    assert s.llm_fallback_model == ""
    assert s.llm_max_prompt_price is None
    assert s.llm_max_completion_price is None
    assert s.llm_provider_sort == ""


def test_invalid_provider_sort_raises(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER_SORT", "cheapest")  # невалидно

    import rag.config as config
    with pytest.raises(RuntimeError):
        config.load_settings()
