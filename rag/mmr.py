"""
mmr.py — Maximal Marginal Relevance.

Зачем нужно: на выходе hybrid+rerank мы получаем кандидатов отсортированных
ПО РЕЛЕВАНТНОСТИ. Часто top-5 оказываются «дублирующимися» — соседние
чанки одного документа, перекрывающиеся фрагменты, разные кусочки одной
мысли. Это вредно: контекст LLM занят повторяющейся информацией, а
другие аспекты вопроса не покрыты.

MMR (Carbonell & Goldstein, 1998) — итеративный алгоритм отбора,
балансирующий **релевантность** и **разнообразие**:

    pick = argmax_c [  λ · relevance(c)  -  (1-λ) · max_sim(c, S)  ]
                       ↑                       ↑
                       насколько c              насколько c уже похож
                       полезен                  на что-то выбранное

  - λ = 1.0  → чистая релевантность (= обычная сортировка по score)
  - λ = 0.0  → чистое разнообразие, релевантность игнорируется
  - λ = 0.5  → баланс (хороший дефолт)

Алгоритм:
  1. Кладём в результат самого релевантного.
  2. Дальше итеративно: из оставшихся берём того, у кого
     λ·rel - (1-λ)·max_sim_to_selected максимально.
  3. Повторяем пока не наберём top_k.

Где similarity? Считаем cosine между эмбеддингами чанков. Эмбеддинги
у нас уже лежат в pgvector — забираем их одним SQL-запросом.
"""

from __future__ import annotations

import math
from dataclasses import replace

from rag.vector_store import RetrievedChunk


# Тип ключа чанка: (source, chunk_index).
ChunkKey = tuple[str, int]


def _cosine(a: list[float], b: list[float]) -> float:
    """Косинусное сходство двух векторов. Дублируется и в server.py,
    но здесь нужно изолированно, чтобы модуль был самостоятельным."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _primary_score(c: RetrievedChunk) -> float:
    """
    Возвращает «основную» меру релевантности чанка — ту, по которой
    его уже отсортировал предыдущий этап.

    Приоритет: reranker_score > rrf_score > similarity (cosine).
    Если был rerank — он точнее, его и используем для MMR.
    """
    if c.reranker_score is not None:
        return c.reranker_score
    if c.rrf_score is not None:
        return c.rrf_score
    return c.similarity or 0.0


def mmr_select(
    candidates: list[RetrievedChunk],
    embeddings: dict[ChunkKey, list[float]],
    top_k: int,
    lambda_: float,
) -> list[RetrievedChunk]:
    """
    Применяет MMR к списку кандидатов, возвращает отобранные top_k
    в порядке выбора (первый — самый релевантный, дальше — с учётом
    разнообразия).

    lambda_ ∈ [0, 1]:
        1.0 — без MMR, чистая релевантность (порядок не меняется).
        0.0 — чистое разнообразие, релевантность игнорируется.
        0.5 — рекомендованный баланс.

    Кандидаты для которых нет эмбеддинга в `embeddings` игнорируются
    (не должны попадаться, но защита от пропуска данных).
    """
    if not candidates or top_k <= 0:
        return []
    # Если lambda=1.0 — MMR деградирует в обычное взятие top_k.
    # Это и так корректно работает по формуле, но кейс достаточно частый
    # чтобы оптимизировать (и сделать поведение очевидным).
    if lambda_ >= 0.999:
        return candidates[:top_k]

    # Нормируем relevance в [0, 1]. Это критично: relevance и similarity
    # должны быть в одной шкале, иначе λ-баланс будет несимметричным.
    # reranker_score уже в [0, 1] (sigmoid), но cos и rrf — нет.
    scores = [_primary_score(c) for c in candidates]
    s_max = max(scores)
    s_min = min(scores)
    span = (s_max - s_min) or 1.0
    norm_rel = [(s - s_min) / span for s in scores]

    n = len(candidates)
    selected_idx: list[int] = []
    remaining = set(range(n))

    # Шаг 1: первый кандидат — просто самый релевантный.
    first = max(remaining, key=lambda i: norm_rel[i])
    selected_idx.append(first)
    remaining.discard(first)

    # Шаг 2+: итеративно добавляем по одному.
    while len(selected_idx) < top_k and remaining:
        best_i = None
        best_score = float("-inf")

        for i in remaining:
            key_i = (candidates[i].source, candidates[i].chunk_index)
            vec_i = embeddings.get(key_i)
            if vec_i is None:
                continue

            # Считаем максимальное сходство с уже выбранными — это «штраф»
            # за похожесть. Чем оно выше, тем больше кандидат теряет.
            max_sim = 0.0
            for s_idx in selected_idx:
                key_s = (candidates[s_idx].source, candidates[s_idx].chunk_index)
                vec_s = embeddings.get(key_s)
                if vec_s is None:
                    continue
                sim = _cosine(vec_i, vec_s)
                if sim > max_sim:
                    max_sim = sim

            score = lambda_ * norm_rel[i] - (1.0 - lambda_) * max_sim
            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break
        selected_idx.append(best_i)
        remaining.discard(best_i)

    # Возвращаем кандидатов в порядке выбора MMR. Каждому проставляем
    # mmr_rank — это позиция в MMR-отборе (1-based), чтобы UI мог
    # показать «#3 → #1» как у reranker'а.
    result: list[RetrievedChunk] = []
    for new_rank, idx in enumerate(selected_idx, start=1):
        result.append(replace(candidates[idx], mmr_rank=new_rank))
    return result
