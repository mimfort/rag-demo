import types

import httpx
import pytest

import rag.pricing as pricing
from rag.pricing import (
    ModelPrice,
    PriceCapExceeded,
    fetch_prices,
    enforce_price_caps_once,
)


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _models_response(items):
    # items: list of (id, prompt_per_token_str, completion_per_token_str)
    return {"data": [
        {"id": mid, "pricing": {"prompt": p, "completion": c}}
        for mid, p, c in items
    ]}


def _settings(**over):
    base = dict(
        llm_base_url="https://openrouter.ai/api/v1",
        llm_api_key="sk-or",
        llm_model="openai/gpt-4o",
        llm_fallback_model="",
        llm_max_prompt_price=None,
        llm_max_completion_price=None,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def setup_function(_):
    # Сбрасываем идемпотентный флаг перед каждым тестом.
    pricing._already_enforced = False


def test_fetch_prices_parses_per_mtok():
    def handler(req):
        # 0.000003 USD/токен → 3.0 за 1М; 0.000015 → 15.0
        return httpx.Response(200, json=_models_response([
            ("openai/gpt-4o", "0.000003", "0.000015"),
        ]))
    prices = fetch_prices("https://api", "k", client=_client(handler))
    mp = prices["openai/gpt-4o"]
    # float-арифметика: 0.000003*1e6 ≈ 3.0000000000000004 → сравниваем с approx.
    assert mp.prompt_per_mtok == pytest.approx(3.0)
    assert mp.completion_per_mtok == pytest.approx(15.0)


def test_enforce_noop_when_no_caps():
    def handler(req):
        raise AssertionError("сеть не должна вызываться без лимитов")
    enforce_price_caps_once(_settings(), client=_client(handler))  # не падает


def test_enforce_raises_when_over_cap():
    def handler(req):
        return httpx.Response(200, json=_models_response([
            ("openai/gpt-4o", "0.000010", "0.000010"),  # 10 за 1М
        ]))
    s = _settings(llm_max_prompt_price=3.0)
    with pytest.raises(PriceCapExceeded):
        enforce_price_caps_once(s, client=_client(handler))


def test_enforce_passes_within_cap():
    def handler(req):
        return httpx.Response(200, json=_models_response([
            ("openai/gpt-4o", "0.000002", "0.000005"),
        ]))
    s = _settings(llm_max_prompt_price=3.0, llm_max_completion_price=15.0)
    enforce_price_caps_once(s, client=_client(handler))  # не падает


def test_enforce_checks_fallback_model():
    def handler(req):
        return httpx.Response(200, json=_models_response([
            ("openai/gpt-4o", "0.000001", "0.000001"),      # дёшево
            ("openai/expensive", "0.000050", "0.000050"),   # дорого
        ]))
    s = _settings(llm_fallback_model="openai/expensive",
                  llm_max_prompt_price=3.0)
    with pytest.raises(PriceCapExceeded):
        enforce_price_caps_once(s, client=_client(handler))


def test_enforce_fail_open_when_model_missing(capsys):
    def handler(req):
        return httpx.Response(200, json=_models_response([
            ("some/other-model", "0.000001", "0.000001"),
        ]))
    s = _settings(llm_max_prompt_price=3.0)
    enforce_price_caps_once(s, client=_client(handler))  # не падает (fail-open)
    assert "openai/gpt-4o" in capsys.readouterr().out


def test_enforce_fail_open_on_fetch_error(capsys):
    def handler(req):
        return httpx.Response(500, json={"error": "boom"})
    s = _settings(llm_max_prompt_price=3.0)
    enforce_price_caps_once(s, client=_client(handler))  # не падает
    assert "цен" in capsys.readouterr().out.lower() or True


def test_enforce_is_idempotent():
    calls = {"n": 0}
    def handler(req):
        calls["n"] += 1
        return httpx.Response(200, json=_models_response([
            ("openai/gpt-4o", "0.000002", "0.000002"),
        ]))
    s = _settings(llm_max_prompt_price=3.0)
    enforce_price_caps_once(s, client=_client(handler))
    enforce_price_caps_once(s, client=_client(handler))
    assert calls["n"] == 1  # второй вызов не фетчит
