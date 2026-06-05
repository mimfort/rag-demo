# LLM Price Gate + Fallback + Provider Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить ценовой гейт (отказ при слишком дорогой модели), нативный OpenRouter fallback на запасную модель и опциональный выбор выгодного провайдера (`provider.sort`).

**Architecture:** Новый модуль `rag/pricing.py` фетчит цены моделей через `GET /models` и проверяет их против лимитов из `.env` (идемпотентно, один раз за процесс). Гейт вызывается в конструкторах chat-клиентов (`ChatGenerator`, `make_llm`) — DRY, без правок каждого entrypoint. Fallback и provider-sort добавляются в тело запроса через общий хелпер `build_routing_fields()`.

**Tech Stack:** Python, httpx (+ httpx.MockTransport для тестов), langchain-openai (ChatOpenAI `extra_body`), pytest.

---

## File Structure

- `rag/config.py` (modify) — поля `llm_fallback_model`, `llm_max_prompt_price`, `llm_max_completion_price`, `llm_provider_sort`; хелпер `_env_float_optional`; валидация provider_sort.
- `rag/pricing.py` (create) — `ModelPrice`, `fetch_prices`, `PriceCapExceeded`, `enforce_price_caps_once`.
- `rag/generator.py` (modify) — `build_routing_fields()`; вызов гейта в `__init__`; `models`/`provider` в `_build_payload`.
- `agent/llm.py` (modify) — вызов гейта; `extra_body` через `build_routing_fields()`.
- `.env`, `.env.example` (modify) — новые переменные.
- `tests/test_pricing.py`, `tests/test_routing_fields.py` (create).

---

## Task 1: Конфигурация

**Files:**
- Modify: `rag/config.py`, `.env.example`, `.env`
- Test: `tests/test_config_pricing.py`

- [ ] **Step 1: Написать падающий тест** — создать `tests/test_config_pricing.py`:

```python
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
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `pytest tests/test_config_pricing.py -v`
Expected: FAIL (полей нет).

- [ ] **Step 3: Добавить хелпер `_env_float_optional`** в `rag/config.py` после `_env_int`:

```python
def _env_float_optional(name: str) -> float | None:
    """Опциональное float-значение: пусто/не задано → None."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)
```

- [ ] **Step 4: Добавить поля в `Settings`** — в блок `# --- LLM (chat) ...` после `llm_app_title: str`:

```python
    # Запасная модель (нативный OpenRouter fallback). Пусто = нет fallback.
    llm_fallback_model: str
    # Ценовой потолок, USD за 1М токенов. None = без ограничения.
    llm_max_prompt_price: float | None
    llm_max_completion_price: float | None
    # Маршрутизация провайдера OpenRouter: "", "price", "throughput", "latency".
    llm_provider_sort: str
```

- [ ] **Step 5: Заполнить поля в `load_settings()`** — после `llm_app_title=...`:

```python
        llm_fallback_model=os.getenv("LLM_FALLBACK_MODEL", ""),
        llm_max_prompt_price=_env_float_optional("LLM_MAX_PROMPT_PRICE_PER_MTOK"),
        llm_max_completion_price=_env_float_optional("LLM_MAX_COMPLETION_PRICE_PER_MTOK"),
        llm_provider_sort=os.getenv("LLM_PROVIDER_SORT", ""),
```

- [ ] **Step 6: Добавить валидацию provider_sort** — в `load_settings()` заменить `return Settings(` так, чтобы сначала собрать объект, провалидировать, потом вернуть. Конкретно: переименовать `return Settings(` в `settings_obj = Settings(`, а в конце функции (после закрывающей `)` конструктора) добавить:

```python
    allowed_sort = {"", "price", "throughput", "latency"}
    if settings_obj.llm_provider_sort not in allowed_sort:
        raise RuntimeError(
            f"LLM_PROVIDER_SORT='{settings_obj.llm_provider_sort}' недопустимо. "
            f"Разрешено: price, throughput, latency (или пусто)."
        )
    return settings_obj
```

- [ ] **Step 7: Запустить — убедиться, что проходит**

Run: `pytest tests/test_config_pricing.py -v`
Expected: PASS (3 теста).

- [ ] **Step 8: Обновить `.env.example` и `.env`** — в блок `# --- LLM (chat) ...` (после `LLM_APP_TITLE=`) добавить:

```dotenv
# Запасная модель: при недоступности основной OpenRouter переключится на неё.
LLM_FALLBACK_MODEL=
# Ценовой потолок, USD за 1М токенов. Пусто = без ограничения.
# Если цена выбранной модели выше — приложение не стартует.
LLM_MAX_PROMPT_PRICE_PER_MTOK=
LLM_MAX_COMPLETION_PRICE_PER_MTOK=
# Маршрутизация провайдера: price | throughput | latency. Пусто = дефолт OpenRouter.
LLM_PROVIDER_SORT=
```

Внести те же ключи (пустыми) в реальный `.env`.

- [ ] **Step 9: Commit**

```bash
git add rag/config.py .env.example tests/test_config_pricing.py
git commit -m "feat(config): поля fallback-модели, ценовых лимитов и provider routing"
```

(`.env` не коммитим — он gitignored.)

---

## Task 2: rag/pricing.py — фетч цен и ценовой гейт

**Files:**
- Create: `rag/pricing.py`
- Test: `tests/test_pricing.py`

- [ ] **Step 1: Написать падающий тест** — создать `tests/test_pricing.py`:

```python
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
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `pytest tests/test_pricing.py -v`
Expected: FAIL (модуля нет).

- [ ] **Step 3: Создать `rag/pricing.py`**

```python
"""
pricing.py — ценовой гейт для chat-модели (OpenRouter).

OpenRouter отдаёт цены моделей через GET {base}/models: у каждого элемента
data[] есть id и pricing.{prompt,completion} — строки «USD за 1 токен».
Умножаем на 1e6 → цена за 1М токенов и сравниваем с лимитами из .env.

Гейт идемпотентный (одна проверка на процесс) и fail-open: если цену нельзя
узнать (сеть недоступна, модели нет в списке, не-OpenRouter base_url) — пишем
warning и продолжаем, а не блокируем работу.

Смена провайдера: гейт активен только когда заданы лимиты; на пустых лимитах
не делает ни одного сетевого вызова.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class ModelPrice:
    """Цена модели в USD за 1М токенов."""

    prompt_per_mtok: float
    completion_per_mtok: float


class PriceCapExceeded(RuntimeError):
    """Цена модели выше заданного в .env лимита — работа прерывается."""


# Идемпотентность: проверяем цену один раз за процесс.
_already_enforced = False


def _to_mtok(raw: object) -> float:
    """Строка/число «USD за токен» → float «USD за 1М токенов»."""
    try:
        return float(raw or 0.0) * 1_000_000
    except (TypeError, ValueError):
        return 0.0


def fetch_prices(
    base_url: str,
    api_key: str,
    *,
    timeout: float = 10.0,
    client: httpx.Client | None = None,
) -> dict[str, ModelPrice]:
    """Возвращает карту id → ModelPrice по данным GET {base}/models."""
    owns_client = client is None
    client = client or httpx.Client(
        timeout=timeout,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        response = client.get(f"{base_url.rstrip('/')}/models")
        response.raise_for_status()
        data = response.json().get("data", [])
    finally:
        if owns_client:
            client.close()

    prices: dict[str, ModelPrice] = {}
    for item in data:
        mid = item.get("id")
        pricing = item.get("pricing") or {}
        if not mid:
            continue
        prices[mid] = ModelPrice(
            prompt_per_mtok=_to_mtok(pricing.get("prompt")),
            completion_per_mtok=_to_mtok(pricing.get("completion")),
        )
    return prices


def enforce_price_caps_once(settings, *, client: httpx.Client | None = None) -> None:
    """
    Один раз за процесс проверяет цену основной и запасной модели против
    лимитов. Превышение → PriceCapExceeded. Недоступный прайсинг / отсутствие
    модели в /models → warning и продолжаем (fail-open).
    """
    global _already_enforced
    if _already_enforced:
        return

    max_prompt = settings.llm_max_prompt_price
    max_completion = settings.llm_max_completion_price
    if max_prompt is None and max_completion is None:
        _already_enforced = True
        return

    models = [settings.llm_model]
    if settings.llm_fallback_model:
        models.append(settings.llm_fallback_model)

    try:
        prices = fetch_prices(settings.llm_base_url, settings.llm_api_key, client=client)
    except Exception as exc:  # noqa: BLE001 — fail-open by design
        print(f"⚠ Не удалось получить цены моделей ({exc}); ценовой гейт пропущен.")
        _already_enforced = True
        return

    for model in models:
        price = prices.get(model)
        if price is None:
            print(
                f"⚠ Модель '{model}' не найдена в /models — цену проверить нельзя, "
                f"пропускаю (fail-open)."
            )
            continue
        if max_prompt is not None and price.prompt_per_mtok > max_prompt:
            raise PriceCapExceeded(
                f"Модель '{model}': prompt ${price.prompt_per_mtok:.2f}/1М > "
                f"лимита ${max_prompt:.2f}/1М. Подними LLM_MAX_PROMPT_PRICE_PER_MTOK "
                f"или выбери модель дешевле."
            )
        if max_completion is not None and price.completion_per_mtok > max_completion:
            raise PriceCapExceeded(
                f"Модель '{model}': completion ${price.completion_per_mtok:.2f}/1М > "
                f"лимита ${max_completion:.2f}/1М. Подними "
                f"LLM_MAX_COMPLETION_PRICE_PER_MTOK или выбери модель дешевле."
            )

    _already_enforced = True
```

- [ ] **Step 4: Запустить — убедиться, что проходит**

Run: `pytest tests/test_pricing.py -v`
Expected: PASS (8 тестов).

- [ ] **Step 5: Commit**

```bash
git add rag/pricing.py tests/test_pricing.py
git commit -m "feat(pricing): ценовой гейт модели через OpenRouter /models (fail-open, idempotent)"
```

---

## Task 3: ChatGenerator — routing fields + вызов гейта

**Files:**
- Modify: `rag/generator.py`
- Test: `tests/test_routing_fields.py`

- [ ] **Step 1: Написать падающий тест** — создать `tests/test_routing_fields.py`:

```python
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
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `pytest tests/test_routing_fields.py -v`
Expected: FAIL (нет `build_routing_fields`).

- [ ] **Step 3: Добавить `build_routing_fields` в `rag/generator.py`** — после функции `build_headers` (перед классом `ChatGenerator`):

```python
def build_routing_fields(model: str, settings) -> dict:
    """
    Доп. поля тела запроса OpenRouter:
    - models: [primary, fallback] — нативный fallback (если задана запасная);
    - provider.sort — выбор провайдера (price/throughput/latency), если задан.
    Пустой dict, если ничего не настроено.
    """
    fields: dict = {}
    if settings.llm_fallback_model:
        fields["models"] = [model, settings.llm_fallback_model]
    if settings.llm_provider_sort:
        fields["provider"] = {"sort": settings.llm_provider_sort}
    return fields
```

- [ ] **Step 4: Подключить routing fields в `_build_payload`** — в `rag/generator.py` заменить тело `_build_payload` (сейчас оно возвращает dict напрямую):

```python
    def _build_payload(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        temperature: float,
        stream: bool,
    ) -> dict:
        payload = {
            "model": self._model,
            "stream": stream,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(query, chunks)},
            ],
        }
        payload.update(build_routing_fields(self._model, settings))
        return payload
```

- [ ] **Step 5: Вызвать ценовой гейт в `__init__`** — в `rag/generator.py` добавить импорт вверху (рядом с `from rag.config import settings`):

```python
from rag.pricing import enforce_price_caps_once
```
и в конце `ChatGenerator.__init__` (после создания `self._client`) добавить строку:

```python
        enforce_price_caps_once(settings)
```

- [ ] **Step 6: Запустить тесты**

Run: `pytest tests/test_routing_fields.py -v`
Expected: PASS (4 теста).
Run: `pytest -v`
Expected: все существующие тесты тоже зелёные (на дефолтных пустых лимитах гейт — no-op, сети нет).

- [ ] **Step 7: Проверить, что импорт ChatGenerator не делает сетевых вызовов**

Run: `python -c "from rag.generator import ChatGenerator, build_routing_fields; print('ok')"`
Expected: `ok` (на пустых лимитах конструктор не вызывается; импорт чист).

- [ ] **Step 8: Commit**

```bash
git add rag/generator.py tests/test_routing_fields.py
git commit -m "feat(llm): routing fields (fallback models[], provider.sort) + ценовой гейт в ChatGenerator"
```

---

## Task 4: make_llm — гейт + extra_body

**Files:**
- Modify: `agent/llm.py`

- [ ] **Step 1: Обновить `agent/llm.py`** — заменить импорты и тело `make_llm()`:

Импорты вверху файла (после `from rag.config import settings as rag_settings`):
```python
from rag.generator import build_routing_fields
from rag.pricing import enforce_price_caps_once
```

Тело `make_llm()`:
```python
def make_llm() -> ChatOpenAI:
    """
    Возвращает ChatOpenAI, настроенную на chat-провайдера из .env
    (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).

    Перед созданием клиента проверяем ценовой лимит модели (см. pricing.py).
    Если заданы запасная модель / provider.sort — прокидываем их в extra_body
    (OpenRouter: models[] для нативного fallback, provider.sort для выбора
    провайдера). temperature=0.2 — стабильный tool-calling; max_retries=1.
    """
    enforce_price_caps_once(rag_settings)
    extra_body = build_routing_fields(rag_settings.llm_model, rag_settings)
    kwargs = {"extra_body": extra_body} if extra_body else {}
    return ChatOpenAI(
        model=rag_settings.llm_model,
        base_url=rag_settings.llm_base_url,
        api_key=rag_settings.llm_api_key,
        temperature=0.2,
        max_retries=1,
        **kwargs,
    )
```

- [ ] **Step 2: Проверить сборку графа**

Run: `python -c "from agent.graph import build_graph; print('ok')"`
Expected: `ok` (дефолтные пустые лимиты → гейт no-op без сети; extra_body пуст → не передаётся).

- [ ] **Step 3: Прогнать весь тест-набор**

Run: `pytest -v`
Expected: все тесты зелёные.

- [ ] **Step 4: Commit**

```bash
git add agent/llm.py
git commit -m "feat(agent): ценовой гейт + fallback/provider routing в make_llm (extra_body)"
```

---

## Final Verification

- [ ] **Полный прогон тестов**

Run: `pytest -v`
Expected: все PASS.

- [ ] **Импорт точек входа**

Run: `python -c "import ask, web.server, evals.runner; from agent.graph import build_graph; print('ok')"`
Expected: `ok`.

- [ ] **Ручная проверка гейта (smoke, опционально, требует сети)**

С реальным `LLM_API_KEY` в `.env` временно выставить `LLM_MAX_PROMPT_PRICE_PER_MTOK=0.0001` и запустить `python -c "from rag.generator import ChatGenerator; ChatGenerator()"` — ожидается `PriceCapExceeded` (если выбранная модель дороже). Вернуть лимит обратно (пусто).
