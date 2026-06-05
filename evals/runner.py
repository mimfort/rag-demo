"""
runner.py — прогоняет голден-сет через разные конфиги retrieval и
печатает сравнительную таблицу.

Конфиги, которые сравниваем:
  1. vector              — только cosine через pgvector
  2. text                — только BM25 через tsvector
  3. hybrid              — vector + text + RRF
  4. hybrid + rerank     — то же + второй проход через cross-encoder

Метрики (см. evals/metrics.py):
  - hit-rate@k     — у скольки запросов в top-K есть хотя бы один правильный
  - recall@k       — доля найденных правильных от всех правильных
  - precision@k    — доля правильных в top-K
  - MRR            — Mean Reciprocal Rank (среднее 1/позиция первого правильного)
  - avg latency    — среднее время одного запроса (без latency на загрузку модели)

Запуск:
  python -m evals.runner                  # все конфиги, k=5
  python -m evals.runner -k 3             # с другим top-k
  python -m evals.runner --skip-rerank    # без cross-encoder (быстрее)
  python -m evals.runner --verbose        # показать поchunk какие нашлись
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Чтобы можно было запускать `python -m evals.runner` из корня проекта.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.embedder import Embedder, make_embedder
from rag.reranker import Reranker, make_reranker
from rag.vector_store import VectorStore
from evals.metrics import (
    AggregateMetrics,
    ChunkKey,
    QueryMetrics,
    aggregate,
    score_query,
)


GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.json"


@dataclass(frozen=True)
class GoldenItem:
    """Один вопрос из голден-сета с разметкой."""

    id: str
    question: str
    tags: list[str]
    relevant: set[ChunkKey]


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[GoldenItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        GoldenItem(
            id=item["id"],
            question=item["question"],
            tags=list(item.get("tags", [])),
            relevant={
                (r["source"], int(r["chunk_index"]))
                for r in item["relevant"]
            },
        )
        for item in raw
    ]


# ----------------------------------------------------------------
# Адаптеры под разные стратегии retrieval — все возвращают
# list[(source, chunk_index)] в порядке убывания релевантности.
# Принимают один embed-вызов через переданный embedder.
# ----------------------------------------------------------------
Retriever = Callable[[str], list[ChunkKey]]


def make_vector_retriever(
    embedder: Embedder,
    store: VectorStore,
    k: int,
) -> Retriever:
    def run(query: str) -> list[ChunkKey]:
        vec = embedder.embed_query(query)
        hits = store.search(vec, top_k=k)
        return [(h.source, h.chunk_index) for h in hits]
    return run


def make_text_retriever(store: VectorStore, k: int) -> Retriever:
    def run(query: str) -> list[ChunkKey]:
        hits = store.text_search(query, top_n=k)
        return [(h.source, h.chunk_index) for h in hits]
    return run


def make_hybrid_retriever(
    embedder: Embedder,
    store: VectorStore,
    k: int,
) -> Retriever:
    def run(query: str) -> list[ChunkKey]:
        vec = embedder.embed_query(query)
        hits = store.hybrid_search(query, vec, top_k=k, candidate_n=20)
        return [(h.source, h.chunk_index) for h in hits]
    return run


def make_hybrid_rerank_retriever(
    embedder: Embedder,
    store: VectorStore,
    reranker: Reranker,
    k: int,
    candidate_n: int = 15,
) -> Retriever:
    def run(query: str) -> list[ChunkKey]:
        vec = embedder.embed_query(query)
        # candidate_n берём НАМНОГО больше k — это смысл reranking.
        cand = store.hybrid_search(query, vec, top_k=candidate_n, candidate_n=20)
        reranked = reranker.rerank(query, cand, top_k=k)
        return [(c.source, c.chunk_index) for c in reranked]
    return run


# ----------------------------------------------------------------
# Прогон одного конфига по всему голден-сету
# ----------------------------------------------------------------
def run_config(
    name: str,
    retriever: Retriever,
    items: list[GoldenItem],
    k: int,
    verbose: bool = False,
) -> tuple[AggregateMetrics, list[tuple[GoldenItem, QueryMetrics]]]:
    per_query: list[QueryMetrics] = []
    latencies: list[float] = []
    details: list[tuple[GoldenItem, QueryMetrics]] = []

    if verbose:
        print(f"\n── {name} " + "─" * (60 - len(name)))

    for item in items:
        t0 = time.perf_counter()
        retrieved = retriever(item.question)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        m = score_query(retrieved, item.relevant, k=k)
        per_query.append(m)
        details.append((item, m))

        if verbose:
            mark = "✓" if m.hit_at_k else "✗"
            pos = ",".join(str(p) for p in m.found_positions) or "—"
            print(f"  {mark} [{item.id}] {item.question}")
            print(f"      позиции правильных: {pos}    RR={m.reciprocal_rank:.3f}")

    return aggregate(per_query, latencies), details


def split_by_tag(
    details: list[tuple[GoldenItem, QueryMetrics]],
    tag: str,
) -> list[QueryMetrics]:
    """Возвращает per-query метрики только для запросов с указанным тегом."""
    return [m for item, m in details if tag in item.tags]


def format_table(rows: list[tuple[str, AggregateMetrics]]) -> str:
    """
    Грубая ASCII-таблица — без зависимостей.
    Колонки: имя конфига, hit@k, recall@k, precision@k, MRR, latency.
    """
    header = ["config", "hit@k", "recall", "precision", "MRR", "ms/query"]
    table = [header]
    for name, m in rows:
        table.append([name, *m.as_row()])

    widths = [max(len(str(c)) for c in col) for col in zip(*table)]
    lines: list[str] = []
    for i, row in enumerate(table):
        line = "  ".join(str(c).ljust(w) for c, w in zip(row, widths))
        lines.append(line)
        if i == 0:
            lines.append("  ".join("─" * w for w in widths))
    return "\n".join(lines)


# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Прогнать голден-сет через все конфиги retrieval.")
    ap.add_argument("-k", "--top-k", type=int, default=5, help="top_k (по умолчанию 5)")
    ap.add_argument("--skip-rerank", action="store_true",
                    help="не запускать конфиг hybrid+rerank (быстрее, без cross-encoder)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="печатать результат каждого запроса")
    args = ap.parse_args()

    items = load_golden_set()
    print(f"Голден-сет: {len(items)} вопросов из {GOLDEN_SET_PATH.name}")
    print(f"top_k = {args.top_k}\n")

    with make_embedder() as embedder, VectorStore() as store:
        rows: list[tuple[str, AggregateMetrics]] = []
        # all_details[config_name] = list[(item, query_metrics)]
        all_details: dict[str, list[tuple[GoldenItem, QueryMetrics]]] = {}

        # Прогоняем 3 быстрых конфига сразу.
        for name, mk in [
            ("vector", lambda: make_vector_retriever(embedder, store, args.top_k)),
            ("text",   lambda: make_text_retriever(store, args.top_k)),
            ("hybrid", lambda: make_hybrid_retriever(embedder, store, args.top_k)),
        ]:
            agg, det = run_config(name, mk(), items, args.top_k, args.verbose)
            rows.append((name, agg))
            all_details[name] = det

        # Конфиг с reranking — отдельный сетевой вызов к Voyage.
        if not args.skip_rerank:
            print("→ инициализирую Voyage reranker…")
            with make_reranker() as reranker:
                retr = make_hybrid_rerank_retriever(
                    embedder, store, reranker, args.top_k,
                )
                agg, det = run_config("hybrid+rerank", retr, items, args.top_k, args.verbose)
                rows.append(("hybrid+rerank", agg))
                all_details["hybrid+rerank"] = det

        print()
        print(format_table(rows))

        # Разбивка по тегам: для каждого тега смотрим MRR каждого конфига.
        # Помогает увидеть «вектор хорош на paraphrase, text — на exact-term».
        all_tags: list[str] = []
        for item in items:
            for t in item.tags:
                if t not in all_tags:
                    all_tags.append(t)

        print("\n— MRR по тегам ─" + "─" * 60)
        header = ["tag", "n"] + list(all_details.keys())
        rows_by_tag = [header]
        for tag in all_tags:
            n = sum(1 for it in items if tag in it.tags)
            row = [tag, str(n)]
            for cfg_name in all_details:
                tag_metrics = split_by_tag(all_details[cfg_name], tag)
                if tag_metrics:
                    mrr = sum(m.reciprocal_rank for m in tag_metrics) / len(tag_metrics)
                    row.append(f"{mrr:.3f}")
                else:
                    row.append("—")
            rows_by_tag.append(row)
        widths = [max(len(str(c)) for c in col) for col in zip(*rows_by_tag)]
        for i, row in enumerate(rows_by_tag):
            print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
            if i == 0:
                print("  ".join("─" * w for w in widths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
