def test_settings_reads_llm_and_voyage(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-or-test")
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-3.5-sonnet")
    monkeypatch.setenv("LLM_HTTP_REFERER", "https://example.com")
    monkeypatch.setenv("LLM_APP_TITLE", "RAG")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    monkeypatch.setenv("VOYAGE_EMBEDDING_MODEL", "voyage-4-large")
    monkeypatch.setenv("VOYAGE_RERANK_MODEL", "rerank-2.5")
    monkeypatch.setenv("EMBEDDING_DIM", "1024")

    import rag.config as config
    s = config.load_settings()

    assert s.llm_base_url == "https://openrouter.ai/api/v1"
    assert s.llm_api_key == "sk-or-test"
    assert s.llm_model == "anthropic/claude-3.5-sonnet"
    assert s.llm_http_referer == "https://example.com"
    assert s.llm_app_title == "RAG"
    assert s.voyage_api_key == "pa-test"
    assert s.voyage_base_url.endswith("voyageai.com/v1")
    assert s.voyage_embedding_model == "voyage-4-large"
    assert s.voyage_rerank_model == "rerank-2.5"
    assert s.embedding_dim == 1024
    assert not hasattr(s, "lm_studio_base_url")
    assert not hasattr(s, "chat_model")
