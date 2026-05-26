"""
metrics.py — стандартные метрики качества retrieval.

В голден-сете каждый вопрос имеет список «правильных» чанков (relevant).
Retrieval-система возвращает список найденных чанков (retrieved), в порядке
убывания релевантности. Сравниваем эти два списка несколькими способами.

Все функции работают с **множествами ключей** — пары (source, chunk_index).
Это устойчиво к тому что чанки внутри RetrievedChunk перетасовываются между
вызовами (например после rerank).
"""

from __future__ import annotations

from dataclasses import dataclass


# Ключ чанка = (source, chunk_index). Этого хватает чтобы уникально идентифицировать.
ChunkKey = tuple[str, int]


@dataclass(frozen=True)
class QueryMetrics:
    """Метрики для одного запроса голден-сета."""

    hit_at_k: bool        # Хотя бы один правильный чанк в top-K?
    recall_at_k: float    # Доля правильных, попавших в top-K  ∈ [0, 1]
    precision_at_k: float  # Доля релевантных среди top-K     ∈ [0, 1]
    reciprocal_rank: float  # 1 / позиция_первого_правильного_в_top_K, иначе 0
    found_positions: list[int]  # 1-based позиции правильных чанков в retrieved


@dataclass(frozen=True)
class AggregateMetrics:
    """Усреднение по всем запросам голден-сета."""

    n_queries: int
    hit_rate: float        # = recall@k но в варианте «есть хотя бы один»
    recall_at_k: float
    precision_at_k: float
    mrr: float             # Mean Reciprocal Rank
    avg_latency_ms: float

    def as_row(self) -> list[str]:
        """Форматированная строка для табличного вывода."""
        return [
            f"{self.hit_rate:.3f}",
            f"{self.recall_at_k:.3f}",
            f"{self.precision_at_k:.3f}",
            f"{self.mrr:.3f}",
            f"{self.avg_latency_ms:.0f}",
        ]


def score_query(
    retrieved: list[ChunkKey],
    relevant: set[ChunkKey],
    k: int,
) -> QueryMetrics:
    """
    Считает все метрики для одного запроса.

    retrieved — что вернула система (упорядоченный список).
    relevant  — какие чанки эксперт пометил как правильные (множество).
    k         — top_k, на котором считаем (обычно 5).
    """
    top_k = retrieved[:k]
    found_positions = [
        i + 1 for i, key in enumerate(top_k) if key in relevant
    ]
    hits = len(found_positions)

    return QueryMetrics(
        hit_at_k=hits > 0,
        # Сколько правильных нашли / сколько всего правильных есть.
        # Не делим на 0 — если в голден-сете нет правильных, считаем 1.0
        # (но таких случаев у нас не должно быть).
        recall_at_k=(hits / len(relevant)) if relevant else 1.0,
        precision_at_k=hits / k if k > 0 else 0.0,
        # MRR использует только позицию ПЕРВОГО правильного результата.
        # Если правильных в top_k не нашлось — 0 (этот запрос «полностью провален»).
        reciprocal_rank=(1.0 / found_positions[0]) if found_positions else 0.0,
        found_positions=found_positions,
    )


def aggregate(
    per_query: list[QueryMetrics],
    latencies_ms: list[float],
) -> AggregateMetrics:
    """Среднее арифметическое по всем запросам."""
    n = len(per_query)
    if n == 0:
        return AggregateMetrics(0, 0, 0, 0, 0, 0)
    return AggregateMetrics(
        n_queries=n,
        hit_rate=sum(1 for m in per_query if m.hit_at_k) / n,
        recall_at_k=sum(m.recall_at_k for m in per_query) / n,
        precision_at_k=sum(m.precision_at_k for m in per_query) / n,
        mrr=sum(m.reciprocal_rank for m in per_query) / n,
        avg_latency_ms=sum(latencies_ms) / max(1, len(latencies_ms)),
    )
