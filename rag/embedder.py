"""
embedder.py — превращает текст в вектор через LM Studio.

Что такое «эмбеддинг»?
Эмбеддинг — это вектор фиксированной длины (у bge-m3 — 1024 числа float),
который кодирует «смысл» текста. Модель обучена так, что близкие по смыслу
тексты получают близкие векторы (по cosine similarity), а несвязанные —
далёкие. Именно это позволяет искать «по смыслу», а не по совпадению слов.

API LM Studio совместимо с OpenAI. Запрос на эмбеддинги выглядит так:

    POST {base_url}/embeddings
    Authorization: Bearer {api_key}
    Content-Type: application/json

    {
      "model": "text-embedding-bge-m3",
      "input": ["текст 1", "текст 2", ...]
    }

Ответ:

    {
      "data": [
        {"embedding": [0.0123, -0.045, ...], "index": 0, "object": "embedding"},
        {"embedding": [...],                  "index": 1, "object": "embedding"}
      ],
      "model": "text-embedding-bge-m3",
      "object": "list",
      "usage": {...}
    }

Здесь мы намеренно НЕ используем библиотеку `openai` — хочется чтобы было
видно сырое HTTP-взаимодействие. В реальном проекте можно взять и SDK.
"""

from __future__ import annotations

from typing import Iterable

import httpx

from rag.config import settings


class LMStudioEmbedder:
    """
    Клиент к /v1/embeddings LM Studio.
    Поддерживает батч (несколько текстов за один запрос) — это быстрее,
    чем по одному, потому что модель умеет считать их параллельно на GPU
    и нет накладных расходов на HTTP per call.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        # Если конкретный параметр не передан — берём из глобальных настроек.
        # Такой паттерн удобен для тестирования: в тестах можно подсунуть моки.
        self._base_url = (base_url or settings.lm_studio_base_url).rstrip("/")
        self._api_key = api_key or settings.lm_studio_api_key
        self._model = model or settings.embedding_model
        # Создаём httpx-клиент один раз: внутри он держит keep-alive
        # connection pool, что заметно ускоряет серию запросов.
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    # --- удобные методы ---

    def embed_one(self, text: str) -> list[float]:
        """Эмбеддинг одного текста — обёртка над embed_many для удобства."""
        vectors = self.embed_many([text])
        return vectors[0]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        """
        Эмбеддинги для пачки текстов.
        Принимаем любую итерируемую коллекцию, но материализуем в list:
        нам нужно знать длину и иметь возможность сериализовать в JSON.
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        # Формируем тело запроса в формате OpenAI Embeddings API.
        payload = {
            "model": self._model,
            "input": texts_list,
        }

        url = f"{self._base_url}/embeddings"
        response = self._client.post(url, json=payload)
        # raise_for_status() выкидывает исключение при 4xx/5xx —
        # пусть лучше упадём громко, чем будем работать с битыми данными.
        response.raise_for_status()
        data = response.json()

        # API не гарантирует порядок элементов в data, поэтому сортируем по index
        # (на всякий случай — в практике LM Studio возвращает по порядку).
        items = sorted(data["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in items]

        # Лёгкая проверка размерности — если модель внезапно отдала векторы
        # другой длины, чем мы ожидаем в схеме БД, лучше упасть сейчас,
        # а не получить ошибку pgvector при INSERT.
        for i, vec in enumerate(vectors):
            if len(vec) != settings.embedding_dim:
                raise RuntimeError(
                    f"Эмбеддинг {i} имеет размерность {len(vec)}, "
                    f"а ожидается {settings.embedding_dim}. "
                    f"Проверь EMBEDDING_DIM в .env и модель в LM Studio."
                )

        return vectors

    def close(self) -> None:
        """Закрывает HTTP-клиент. Вызывать в конце работы."""
        self._client.close()

    # Поддержка `with LMStudioEmbedder() as emb:` — гарантированно закроет клиент.
    def __enter__(self) -> "LMStudioEmbedder":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
