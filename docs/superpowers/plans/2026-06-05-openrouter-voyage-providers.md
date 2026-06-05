# OpenRouter + Voyage AI Providers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Перевести проект с локальной LM Studio на OpenRouter (chat-LLM) и Voyage AI (embeddings + rerank), сохранив стриминг и провайдер-агностичность.

**Architecture:** Chat использует OpenAI-совместимый транспорт (OpenRouter и LM Studio = одна реализация, разница в `.env`). Embeddings и rerank вынесены за `Protocol`-интерфейсы с реализациями `VoyageEmbedder` / `VoyageReranker` и фабриками `make_embedder()` / `make_reranker()` — единственными точками выбора провайдера. Локальный cross-encoder и его зависимости (torch/sentence-transformers) удаляются.

**Tech Stack:** Python, httpx (сырой HTTP к Voyage и OpenAI-совместимому chat), langchain-openai (`ChatOpenAI` для агента), pytest + httpx.MockTransport для тестов.

---

## File Structure

- `rag/config.py` (modify) — поля `llm_*` и `voyage_*` вместо `lm_studio_*`/`chat_model`.
- `.env.example`, `.env` (modify) — новые переменные.
- `requirements.txt` (modify) — убрать `sentence-transformers`, добавить `pytest`.
- `rag/generator.py` (modify) — `LMStudioGenerator` → `ChatGenerator`, заголовки атрибуции, чтение `llm_*`.
- `agent/llm.py` (modify) — `make_llm()` читает `llm_*`.
- `rag/embedder.py` (rewrite) — `Protocol Embedder`, `VoyageEmbedder`, `make_embedder()`.
- `rag/reranker.py` (rewrite) — `Protocol Reranker`, `VoyageReranker`, `make_reranker()`.
- Call-sites (modify): `ask.py`, `web/server.py`, `rag/rewriter.py`, `rag/router.py`, `rag/decomposer.py`, `rag/history.py`, `rag/retriever.py`, `ingest.py`, `evals/runner.py`.
- `tests/` (create) — `test_config.py`, `test_chat_generator.py`, `test_voyage_embedder.py`, `test_voyage_reranker.py`.
- `README.md` (modify) — заметка про переиндексацию.

---

## Task 1: Зависимости и тестовый каркас

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`

- [ ] **Step 1: Добавить pytest, убрать sentence-transformers**

В `requirements.txt` удалить весь блок про reranker (`sentence-transformers>=3.0,<5.0` и комментарии к нему). В конец файла добавить:

```
# --- Тесты ---
# pytest + httpx.MockTransport: юнит-тесты провайдеров без реальной сети.
pytest>=8.0,<9.0
```

- [ ] **Step 2: Создать пакет tests**

Создать пустой файл `tests/__init__.py` (содержимое — одна строка комментария):

```python
# Юнит-тесты провайдеров (config, chat, embeddings, rerank).
```

- [ ] **Step 3: Установить зависимости**

Run: `pip install -r requirements.txt`
Expected: pytest установлен, ошибок нет.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt tests/__init__.py
git commit -m "chore: добавить pytest, убрать sentence-transformers из зависимостей"
```

---

## Task 2: Конфигурация (Settings + .env)

**Files:**
- Modify: `rag/config.py`
- Modify: `.env.example`, `.env`
- Test: `tests/test_config.py`

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_config.py`:

```python
import importlib
import os


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
    # Старых полей быть не должно.
    assert not hasattr(s, "lm_studio_base_url")
    assert not hasattr(s, "chat_model")
```

- [ ] **Step 2: Запустить тест — убедиться, что падает**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (поля `llm_base_url`/`voyage_*` ещё не существуют).

- [ ] **Step 3: Обновить Settings**

В `rag/config.py` в `@dataclass class Settings` заменить блок `# --- LM Studio ---`:

```python
    # --- LLM (chat) — OpenAI-совместимый провайдер (OpenRouter по умолчанию) ---
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    # Опциональные заголовки атрибуции OpenRouter (пустая строка = не слать).
    llm_http_referer: str
    llm_app_title: str

    # --- Voyage AI (embeddings + rerank) ---
    voyage_api_key: str
    voyage_base_url: str
    voyage_embedding_model: str
    voyage_rerank_model: str
    embedding_dim: int
```

(Старые поля `lm_studio_base_url`, `lm_studio_api_key`, `embedding_model`, `embedding_dim`, `chat_model` из этого блока удалить — `embedding_dim` перенесён в блок Voyage выше.)

- [ ] **Step 4: Обновить load_settings**

В `rag/config.py` в `load_settings()` заменить строки чтения LM Studio / embedding / chat на:

```python
        llm_base_url=_env("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        llm_api_key=_env("LLM_API_KEY"),
        llm_model=_env("LLM_MODEL"),
        llm_http_referer=os.getenv("LLM_HTTP_REFERER", ""),
        llm_app_title=os.getenv("LLM_APP_TITLE", ""),
        voyage_api_key=_env("VOYAGE_API_KEY"),
        voyage_base_url=_env("VOYAGE_BASE_URL", "https://api.voyageai.com/v1"),
        voyage_embedding_model=_env("VOYAGE_EMBEDDING_MODEL", "voyage-4-large"),
        voyage_rerank_model=_env("VOYAGE_RERANK_MODEL", "rerank-2.5"),
        embedding_dim=_env_int("EMBEDDING_DIM", 1024),
```

- [ ] **Step 5: Запустить тест — убедиться, что проходит**

Run: `pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 6: Обновить .env.example и .env**

В `.env.example` заменить блок LM Studio и RAG-параметры провайдеров на:

```dotenv
# --- LLM (chat) — OpenAI-совместимый провайдер ---
# По умолчанию OpenRouter. Для другого оператора поменяй base_url/ключ/модель.
LLM_BASE_URL=https://openrouter.ai/api/v1
# Токен OpenRouter (Settings → Keys), передаётся как Authorization: Bearer <ключ>.
LLM_API_KEY=your-openrouter-api-key-here
# Идентификатор модели у оператора, напр. anthropic/claude-3.5-sonnet или openai/gpt-4o.
LLM_MODEL=
# Опциональные заголовки атрибуции OpenRouter (можно оставить пустыми).
LLM_HTTP_REFERER=
LLM_APP_TITLE=

# --- Voyage AI (embeddings + rerank) ---
VOYAGE_API_KEY=your-voyage-api-key-here
VOYAGE_BASE_URL=https://api.voyageai.com/v1
# Embedding-модель. voyage-4-large по умолчанию даёт 1024 измерения.
VOYAGE_EMBEDDING_MODEL=voyage-4-large
# Reranker-модель.
VOYAGE_RERANK_MODEL=rerank-2.5
# Размерность векторов. Должна совпадать с vector(1024) в db/init.sql.
EMBEDDING_DIM=1024
```

В реальном `.env` внести те же ключи (вставив фактические токены пользователя там, где он их укажет; `LLM_MODEL` оставить для заполнения пользователем).

- [ ] **Step 7: Commit**

```bash
git add rag/config.py .env.example tests/test_config.py
git commit -m "feat(config): провайдер-нейтральные LLM_* и VOYAGE_* настройки"
```

---

## Task 3: ChatGenerator (OpenRouter, переименование, заголовки)

**Files:**
- Modify: `rag/generator.py`
- Test: `tests/test_chat_generator.py`

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_chat_generator.py`:

```python
from rag.generator import build_headers


def test_build_headers_minimal():
    h = build_headers("sk-or-x")
    assert h["Authorization"] == "Bearer sk-or-x"
    assert h["Content-Type"] == "application/json"
    assert "HTTP-Referer" not in h
    assert "X-Title" not in h


def test_build_headers_with_attribution():
    h = build_headers("sk-or-x", referer="https://app.test", title="RAG")
    assert h["HTTP-Referer"] == "https://app.test"
    assert h["X-Title"] == "RAG"
```

- [ ] **Step 2: Запустить тест — убедиться, что падает**

Run: `pytest tests/test_chat_generator.py -v`
Expected: FAIL (нет `build_headers`).

- [ ] **Step 3: Добавить build_headers и переименовать класс**

В `rag/generator.py`:

1. Добавить функцию перед классом:

```python
def build_headers(
    api_key: str, referer: str = "", title: str = ""
) -> dict[str, str]:
    """
    Заголовки для OpenAI-совместимого chat-API.
    referer/title — необязательная атрибуция OpenRouter (рейтинги приложений);
    пустые значения не отправляем.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers
```

2. Переименовать `class LMStudioGenerator:` → `class ChatGenerator:` и обновить докстринг:

```python
class ChatGenerator:
    """
    Клиент к /chat/completions OpenAI-совместимого провайдера (OpenRouter
    по умолчанию; подойдёт любой OpenAI-compat endpoint — достаточно сменить
    LLM_BASE_URL/LLM_API_KEY/LLM_MODEL в .env).
    """
```

3. В `__init__` заменить чтение настроек и сборку заголовков:

```python
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = (base_url or settings.llm_base_url).rstrip("/")
        self._api_key = api_key or settings.llm_api_key
        self._model = model or settings.llm_model
        self._timeout = timeout
        self._client = httpx.Client(
            timeout=timeout,
            headers=build_headers(
                self._api_key,
                settings.llm_http_referer,
                settings.llm_app_title,
            ),
        )
```

4. Обновить `__enter__` тайп-хинт: `def __enter__(self) -> "ChatGenerator":`.

- [ ] **Step 4: Запустить тест — убедиться, что проходит**

Run: `pytest tests/test_chat_generator.py -v`
Expected: PASS.

- [ ] **Step 5: Обновить импорты ChatGenerator во всех потребителях**

Заменить `LMStudioGenerator` → `ChatGenerator` (импорт + использования + тайп-хинты) в файлах:
- `ask.py` (строки 27, 53)
- `web/server.py` (строки 52, 126, 149)
- `rag/rewriter.py` (строки 34, 85, 88)
- `rag/router.py` (строки 36, 100)
- `rag/decomposer.py` (строки 34, 96)
- `rag/history.py` (строки 35, 200)

Команда для проверки, что не осталось упоминаний:

Run: `grep -rn --include='*.py' LMStudioGenerator . | grep -v .venv`
Expected: пусто.

- [ ] **Step 6: Обновить verbose-вывод в ask.py**

В `ask.py` заменить `settings.embedding_model` → `settings.voyage_embedding_model` и `settings.chat_model` → `settings.llm_model` (в f-строках verbose-вывода, ~строки 58 и 65).

- [ ] **Step 7: Запустить весь тест-набор**

Run: `pytest -v`
Expected: PASS (config + chat generator).

- [ ] **Step 8: Commit**

```bash
git add rag/generator.py ask.py web/server.py rag/rewriter.py rag/router.py rag/decomposer.py rag/history.py tests/test_chat_generator.py
git commit -m "feat(llm): ChatGenerator на OpenRouter (OpenAI-compat) + заголовки атрибуции"
```

---

## Task 4: agent/llm.py на OpenRouter

**Files:**
- Modify: `agent/llm.py`

- [ ] **Step 1: Обновить make_llm**

В `agent/llm.py` заменить тело `make_llm()` и докстринг модуля (упоминание LM Studio → OpenAI-совместимый провайдер):

```python
    return ChatOpenAI(
        model=rag_settings.llm_model,
        base_url=rag_settings.llm_base_url,
        api_key=rag_settings.llm_api_key,
        temperature=0.2,
        max_retries=1,
    )
```

- [ ] **Step 2: Проверить, что граф собирается**

Run: `python -c "from agent.graph import build_graph; print('ok')"`
Expected: печатает `ok` без ошибок импорта/конфига (при заполненном `.env`).

- [ ] **Step 3: Commit**

```bash
git add agent/llm.py
git commit -m "feat(agent): make_llm() читает LLM_* (OpenRouter)"
```

---

## Task 5: VoyageEmbedder (Protocol + реализация + фабрика)

**Files:**
- Rewrite: `rag/embedder.py`
- Test: `tests/test_voyage_embedder.py`

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_voyage_embedder.py`:

```python
import httpx
import pytest

from rag.embedder import VoyageEmbedder, make_embedder, Embedder


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_embed_documents_parses_and_sorts():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["body"] = json.loads(request.content)
        # Намеренно перепутанный порядок index — клиент должен отсортировать.
        return httpx.Response(200, json={"data": [
            {"index": 1, "embedding": [0.1, 0.2, 0.3]},
            {"index": 0, "embedding": [0.4, 0.5, 0.6]},
        ]})

    emb = VoyageEmbedder(api_key="pa", model="voyage-4-large", dim=3,
                         client=_client(handler))
    vectors = emb.embed_documents(["a", "b"])

    assert vectors == [[0.4, 0.5, 0.6], [0.1, 0.2, 0.3]]
    assert captured["body"]["input_type"] == "document"
    assert captured["body"]["model"] == "voyage-4-large"


def test_embed_query_sets_input_type_query():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [
            {"index": 0, "embedding": [1.0, 2.0, 3.0]},
        ]})

    emb = VoyageEmbedder(api_key="pa", dim=3, client=_client(handler))
    vec = emb.embed_query("hello")

    assert vec == [1.0, 2.0, 3.0]
    assert captured["body"]["input_type"] == "query"
    assert captured["body"]["input"] == ["hello"]


def test_embed_dimension_mismatch_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [
            {"index": 0, "embedding": [1.0, 2.0]},  # длина 2, ждём 3
        ]})

    emb = VoyageEmbedder(api_key="pa", dim=3, client=_client(handler))
    with pytest.raises(RuntimeError):
        emb.embed_query("x")


def test_empty_documents_no_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("сеть не должна вызываться на пустом входе")

    emb = VoyageEmbedder(api_key="pa", dim=3, client=_client(handler))
    assert emb.embed_documents([]) == []


def test_make_embedder_returns_protocol(monkeypatch):
    monkeypatch.setattr("rag.embedder.settings",
                        type("S", (), {"voyage_base_url": "https://api.voyageai.com/v1",
                                       "voyage_api_key": "pa",
                                       "voyage_embedding_model": "voyage-4-large",
                                       "embedding_dim": 1024})())
    e = make_embedder()
    assert isinstance(e, Embedder)
```

- [ ] **Step 2: Запустить тест — убедиться, что падает**

Run: `pytest tests/test_voyage_embedder.py -v`
Expected: FAIL (`VoyageEmbedder`/`make_embedder`/`Embedder` не существуют).

- [ ] **Step 3: Переписать rag/embedder.py**

Полностью заменить содержимое `rag/embedder.py`:

```python
"""
embedder.py — превращает текст в вектор через Voyage AI.

Эмбеддинг — вектор фиксированной длины (voyage-4-large по умолчанию 1024
числа float), кодирующий «смысл» текста: близкие по смыслу тексты получают
близкие векторы (cosine), что позволяет искать «по смыслу».

Voyage-специфика: параметр input_type. Для ретрива качество выше, если при
индексации слать input_type="document", а на поиске — input_type="query"
(модель подмешивает разные служебные префиксы). Поэтому интерфейс разделён
на embed_documents и embed_query.

API Voyage:
    POST {base}/embeddings
    {"model": "...", "input": ["...", ...], "input_type": "document"|"query"}
ответ: {"data": [{"index": 0, "embedding": [...]}, ...], ...}

Смена провайдера: реализовать класс под Protocol Embedder и вернуть его из
make_embedder() — остальной код зависит только от интерфейса.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

import httpx

from rag.config import settings


@runtime_checkable
class Embedder(Protocol):
    """Контракт эмбеддера. Реализации не обязаны наследоваться явно."""

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]: ...


class VoyageEmbedder:
    """
    Клиент к /embeddings Voyage. Батч (несколько текстов за запрос, ≤ 1000)
    быстрее поштучной отправки: нет накладных расходов на HTTP per call.

    client можно подменить (httpx.MockTransport) для тестов без сети.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        dim: int | None = None,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = (base_url or settings.voyage_base_url).rstrip("/")
        self._api_key = api_key or settings.voyage_api_key
        self._model = model or settings.voyage_embedding_model
        self._dim = dim if dim is not None else settings.embedding_dim
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        texts_list = list(texts)
        if not texts_list:
            return []
        return self._embed(texts_list, "document")

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        payload = {
            "model": self._model,
            "input": texts,
            "input_type": input_type,
        }
        response = self._client.post(f"{self._base_url}/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()

        items = sorted(data["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in items]

        for i, vec in enumerate(vectors):
            if len(vec) != self._dim:
                raise RuntimeError(
                    f"Эмбеддинг {i} имеет размерность {len(vec)}, "
                    f"а ожидается {self._dim}. Проверь EMBEDDING_DIM в .env "
                    f"и VOYAGE_EMBEDDING_MODEL."
                )
        return vectors

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "VoyageEmbedder":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def make_embedder() -> Embedder:
    """Единственная точка выбора провайдера эмбеддингов."""
    return VoyageEmbedder()
```

- [ ] **Step 4: Запустить тест — убедиться, что проходит**

Run: `pytest tests/test_voyage_embedder.py -v`
Expected: PASS (5 тестов).

- [ ] **Step 5: Commit**

```bash
git add rag/embedder.py tests/test_voyage_embedder.py
git commit -m "feat(embed): VoyageEmbedder за Protocol-интерфейсом + make_embedder()"
```

---

## Task 6: Обновить потребителей эмбеддера

**Files:**
- Modify: `rag/retriever.py`, `ingest.py`, `evals/runner.py`

- [ ] **Step 1: retriever.py — embed_query + тип Embedder**

В `rag/retriever.py`:
- Строка 18: `from rag.embedder import LMStudioEmbedder` → `from rag.embedder import Embedder`.
- Строка 27: тайп-хинт `embedder: LMStudioEmbedder` → `embedder: Embedder`.
- Строка 40: `query_vec = self._embedder.embed_one(query)` → `query_vec = self._embedder.embed_query(query)`.

- [ ] **Step 2: ingest.py — embed_documents + make_embedder**

В `ingest.py`:
- Строка 27: `from rag.embedder import LMStudioEmbedder` → `from rag.embedder import Embedder, make_embedder`.
- Строка 54: тайп-хинт `embedder: LMStudioEmbedder` → `embedder: Embedder`.
- Строка 88: `vectors = embedder.embed_many(texts)` → `vectors = embedder.embed_documents(texts)`.
- Строка 134: `with LMStudioEmbedder() as embedder, VectorStore() as store:` → `with make_embedder() as embedder, VectorStore() as store:`.

(Примечание: `make_embedder()` возвращает `VoyageEmbedder`, у которого есть `__enter__/__exit__`, так что `with` работает.)

- [ ] **Step 3: evals/runner.py — embed_query + make_embedder**

В `evals/runner.py`:
- Строка 38: `from rag.embedder import LMStudioEmbedder` → `from rag.embedder import Embedder, make_embedder`.
- Строки 88, 107, 119: тайп-хинты `embedder: LMStudioEmbedder` → `embedder: Embedder`.
- Строки 93, 112, 126: `vec = embedder.embed_one(query)` → `vec = embedder.embed_query(query)`.
- Строка 213: `with LMStudioEmbedder() as embedder, VectorStore() as store:` → `with make_embedder() as embedder, VectorStore() as store:`.

- [ ] **Step 4: Проверить, что не осталось старого API**

Run: `grep -rn --include='*.py' -E 'LMStudioEmbedder|embed_one|embed_many' . | grep -v .venv`
Expected: пусто.

- [ ] **Step 5: Проверить импорты**

Run: `python -c "import ingest, rag.retriever, evals.runner; print('ok')"`
Expected: печатает `ok`.

- [ ] **Step 6: Commit**

```bash
git add rag/retriever.py ingest.py evals/runner.py
git commit -m "refactor(embed): потребители на Embedder/make_embedder (query/document)"
```

---

## Task 7: VoyageReranker (Protocol + реализация + фабрика)

**Files:**
- Rewrite: `rag/reranker.py`
- Test: `tests/test_voyage_reranker.py`

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_voyage_reranker.py`:

```python
import httpx

from rag.reranker import VoyageReranker, make_reranker, Reranker
from rag.vector_store import RetrievedChunk


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _chunk(content: str) -> RetrievedChunk:
    # Заполняем только обязательные поля; остальные — дефолтные/None.
    return RetrievedChunk(
        source="s", chunk_index=0, content=content, similarity=0.0,
    )


def test_rerank_maps_index_and_sorts():
    def handler(request: httpx.Request) -> httpx.Response:
        # Воспроизводим формат Voyage: index указывает на исходный документ.
        return httpx.Response(200, json={"data": [
            {"index": 0, "relevance_score": 0.2},
            {"index": 1, "relevance_score": 0.9},
        ]})

    rr = VoyageReranker(api_key="pa", model="rerank-2.5", client=_client(handler))
    chunks = [_chunk("low"), _chunk("high")]
    out = rr.rerank("q", chunks)

    assert [c.content for c in out] == ["high", "low"]
    assert out[0].reranker_score == 0.9
    assert out[0].original_rank == 2  # был вторым во входе (index 1 → rank 2)
    assert out[1].reranker_score == 0.2


def test_rerank_applies_top_k_in_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [
            {"index": 0, "relevance_score": 0.5},
        ]})

    rr = VoyageReranker(api_key="pa", client=_client(handler))
    rr.rerank("q", [_chunk("a"), _chunk("b")], top_k=1)
    assert captured["body"]["top_k"] == 1
    assert captured["body"]["query"] == "q"
    assert captured["body"]["documents"] == ["a", "b"]


def test_rerank_empty_no_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("сеть не должна вызываться на пустом входе")

    rr = VoyageReranker(api_key="pa", client=_client(handler))
    assert rr.rerank("q", []) == []


def test_make_reranker_returns_protocol(monkeypatch):
    monkeypatch.setattr("rag.reranker.settings",
                        type("S", (), {"voyage_base_url": "https://api.voyageai.com/v1",
                                       "voyage_api_key": "pa",
                                       "voyage_rerank_model": "rerank-2.5"})())
    assert isinstance(make_reranker(), Reranker)
```

- [ ] **Step 2: Запустить тест — убедиться, что падает**

Run: `pytest tests/test_voyage_reranker.py -v`
Expected: FAIL (нет `VoyageReranker`/`make_reranker`/`Reranker`).

- [ ] **Step 3: Переписать rag/reranker.py**

Полностью заменить содержимое `rag/reranker.py`:

```python
"""
reranker.py — cross-encoder reranking через Voyage AI (/rerank).

Второй этап поиска: на retrieve берём top-15-20 быстрым bi-encoder/BM25,
затем reranker переранжирует только их. Voyage считает релевантность пары
(query, document) на стороне API — модель видит query и document вместе,
что точнее cosine по независимым векторам.

API Voyage:
    POST {base}/rerank
    {"model": "rerank-2.5", "query": "...", "documents": ["...", ...], "top_k": K}
ответ: {"data": [{"index": 0, "relevance_score": 0.93}, ...], ...}
        (index — позиция документа во входном списке; data отсортирован по
         убыванию relevance_score)

Смена провайдера: реализовать класс под Protocol Reranker и вернуть его из
make_reranker().
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol, runtime_checkable

import httpx

from rag.config import settings
from rag.vector_store import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    """Контракт реранкера."""

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]: ...


class VoyageReranker:
    """
    Pointwise reranker через Voyage /rerank.

    client можно подменить (httpx.MockTransport) для тестов без сети.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = (base_url or settings.voyage_base_url).rstrip("/")
        self._api_key = api_key or settings.voyage_api_key
        self._model = model or settings.voyage_rerank_model
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def model_name(self) -> str:
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Шлёт пары (query, chunk.content) в Voyage, проставляет reranker_score
        и original_rank (позиция ДО переранжирования, 1-based), сортирует по
        убыванию score. top_k (если задан) уходит в запрос — API вернёт только
        top_k лучших.
        """
        if not chunks:
            return []

        documents = [c.content for c in chunks]
        payload: dict = {
            "model": self._model,
            "query": query,
            "documents": documents,
        }
        if top_k is not None:
            payload["top_k"] = top_k

        response = self._client.post(f"{self._base_url}/rerank", json=payload)
        response.raise_for_status()
        results = response.json()["data"]

        enriched = [
            replace(
                chunks[r["index"]],
                reranker_score=float(r["relevance_score"]),
                original_rank=r["index"] + 1,
            )
            for r in results
        ]
        enriched.sort(key=lambda c: (c.reranker_score or 0.0), reverse=True)
        return enriched

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "VoyageReranker":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def make_reranker() -> Reranker:
    """Единственная точка выбора провайдера реранкинга."""
    return VoyageReranker()
```

- [ ] **Step 4: Запустить тест — убедиться, что проходит**

Run: `pytest tests/test_voyage_reranker.py -v`
Expected: PASS (4 теста).

- [ ] **Step 5: Commit**

```bash
git add rag/reranker.py tests/test_voyage_reranker.py
git commit -m "feat(rerank): VoyageReranker за Protocol-интерфейсом + make_reranker()"
```

---

## Task 8: Обновить потребителей реранкера

**Files:**
- Modify: `web/server.py`, `evals/runner.py`

- [ ] **Step 1: web/server.py — тип и фабрика**

В `web/server.py`:
- Строка 56: `from rag.reranker import CrossEncoderReranker` → `from rag.reranker import Reranker, make_reranker`.
- Строка 127: `_reranker: CrossEncoderReranker | None = None` → `_reranker: Reranker | None = None`.
- В `_startup()` (строка ~159) заменить:

```python
    try:
        _reranker = make_reranker()
    except Exception as exc:
        # Не критично для остального API — просто rerank будет недоступен.
        print(f"⚠ Reranker не инициализирован: {exc}")
        _reranker = None
```

- Обновить докстринг `_startup` (строки ~142-145): убрать упоминание скачивания модели ~570 МБ с HuggingFace — теперь rerank сетевой (Voyage), инициализация мгновенна, try/except остаётся на случай недоступности сети/ключа.

- [ ] **Step 2: evals/runner.py — тип и фабрика**

В `evals/runner.py`:
- Строка 39: `from rag.reranker import CrossEncoderReranker` → `from rag.reranker import Reranker, make_reranker`.
- Строка 121: тайп-хинт `reranker: CrossEncoderReranker` → `reranker: Reranker`.
- Строка 230: убрать/обновить print `→ загружаю cross-encoder (bge-reranker-v2-m3)…` на `→ инициализирую Voyage reranker…`.
- Строка 231: `with CrossEncoderReranker() as reranker:` → `with make_reranker() as reranker:`.

- [ ] **Step 3: Проверить, что не осталось старого класса**

Run: `grep -rn --include='*.py' -E 'CrossEncoderReranker|sentence_transformers' . | grep -v .venv`
Expected: пусто.

- [ ] **Step 4: Проверить импорты**

Run: `python -c "import web.server, evals.runner; print('ok')"`
Expected: печатает `ok`.

- [ ] **Step 5: Прогнать весь тест-набор**

Run: `pytest -v`
Expected: PASS (config, chat, embedder, reranker).

- [ ] **Step 6: Commit**

```bash
git add web/server.py evals/runner.py
git commit -m "refactor(rerank): потребители на Reranker/make_reranker (Voyage)"
```

---

## Task 9: README — заметка о переиндексации

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Добавить заметку**

В `README.md` найти раздел про настройку/ingest и добавить блок:

```markdown
> **Смена embedding-провайдера требует переиндексации.** Векторы разных
> моделей несовместимы. После перехода на Voyage (`voyage-4-large`) очисти
> таблицу чанков и прогони `python ingest.py` заново. Параметры провайдеров
> задаются в `.env`: `LLM_*` (chat через OpenRouter), `VOYAGE_*` (embeddings
> и rerank).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: заметка о переиндексации при смене embedding-провайдера"
```

---

## Final Verification

- [ ] **Полный прогон тестов**

Run: `pytest -v`
Expected: все тесты PASS.

- [ ] **Импорт всех точек входа**

Run: `python -c "import ask, ingest, web.server, evals.runner; from agent.graph import build_graph; print('ok')"`
Expected: `ok`.

- [ ] **Грепы чистоты (всё пусто)**

Run: `grep -rn --include='*.py' -E 'LMStudioGenerator|LMStudioEmbedder|CrossEncoderReranker|embed_one|embed_many|sentence_transformers|lm_studio|chat_model' . | grep -v .venv`
Expected: пусто.
```

(Опционально, при наличии ключей в `.env` — ручная проверка: запустить `python ask.py "вопрос"` после переиндексации и убедиться, что стриминг работает.)
