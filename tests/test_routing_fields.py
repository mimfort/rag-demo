import types

from rag.generator import build_routing_fields


def _settings(**over):
    base = dict(llm_fallback_model="", llm_provider_sort="")
    base.update(over)
    return types.SimpleNamespace(**base)


def test_routing_empty_by_default():
    assert build_routing_fields("openai/gpt-4o", _settings()) == {}


def test_routing_adds_models_when_fallback_set():
    s = _settings(llm_fallback_model="openai/gpt-4o-mini")
    assert build_routing_fields("openai/gpt-4o", s) == {
        "models": ["openai/gpt-4o", "openai/gpt-4o-mini"],
    }


def test_routing_adds_provider_when_sort_set():
    s = _settings(llm_provider_sort="price")
    assert build_routing_fields("openai/gpt-4o", s) == {
        "provider": {"sort": "price"},
    }


def test_routing_combines_both():
    s = _settings(llm_fallback_model="m2", llm_provider_sort="latency")
    assert build_routing_fields("m1", s) == {
        "models": ["m1", "m2"],
        "provider": {"sort": "latency"},
    }
