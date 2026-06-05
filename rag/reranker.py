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
