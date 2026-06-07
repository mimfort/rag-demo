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

    def close(self) -> None: ...


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
        # упростил: единый путь эмбеддинга
        result = self._embed([text], "document")
        return result[0]

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
