"""
web/server.py — простой FastAPI-сервер поверх нашего RAG.

Что отдаёт:
  GET  /                   — HTML-страница (web/static/index.html)
  GET  /api/health         — { ok: true } для проверки что сервер жив
  POST /api/ask            — обычный JSON-запрос/ответ
                             body: { "query": "...", "top_k": 5 }
                             resp: { "chunks": [...], "prompt": "...", "answer": "..." }
  GET  /api/ask/stream     — Server-Sent Events (для streaming ответа)
                             query params: ?query=...&top_k=5
                             event "meta"  — найденные чанки и итоговый промпт
                             event "token" — очередной кусок ответа от LLM
                             event "done"  — финал

Почему SSE, а не WebSocket?
  Нам нужен только односторонний поток server → client (токены ответа).
  SSE проще: обычный HTTP-ответ с media_type="text/event-stream",
  в браузере читается классом EventSource из коробки.

Запуск:
  uvicorn web.server:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.chunker import chunk_text
from rag.loaders import (
    MAX_UPLOAD_BYTES,
    SUPPORTED_EXTENSIONS,
    UnsupportedFile,
    load_text,
)
from rag.vector_store import ChunkInput

from rag.config import settings
from rag.embedder import LMStudioEmbedder
from rag.generator import ChatGenerator, build_user_prompt
from rag.decomposer import QueryDecomposer
from rag.history import ChatHistoryStore, Message, make_standalone_query
from rag.mmr import mmr_select
from rag.reranker import CrossEncoderReranker
from rag.rewriter import QueryRewriter
from rag.router import QueryRouter, RouteDecision
from rag.vector_store import RetrievedChunk, ScoredChunk, VectorStore, RRF_K
from agent.runner import run_collect, run_stream
from evals.metrics import ChunkKey, QueryMetrics, aggregate, score_query
from evals.runner import (
    GoldenItem,
    load_golden_set,
    make_hybrid_rerank_retriever,
    make_hybrid_retriever,
    make_text_retriever,
    make_vector_retriever,
    split_by_tag,
)


# Допустимые режимы поиска. Используется и в API-моделях, и в логике.
SEARCH_MODES = ("vector", "text", "hybrid")

# Сколько кандидатов вытаскивать на retrieve-этапе, когда reranking включён.
# Reranking перемешивает позиции, поэтому имеет смысл взять заметно больше
# чем top_k — иначе кандидат, который изначально был на 12-м месте, никогда
# не попадёт в финал, даже если он на самом деле лучший.
RERANK_CANDIDATE_N = 15

# Минимальный размер «пула» для MMR — чтобы было из чего выбирать
# с учётом разнообразия. На маленьких пулах MMR не даёт эффекта.
MMR_POOL_SIZE = 15


# ---------------------------------------------------------------------------
# Маленькие математические утилиты для образовательного режима.
# Они не используются в обычном поиске (это делает Postgres), а нужны
# только чтобы показать пользователю РАСЧЁТ cosine similarity «руками».
# ---------------------------------------------------------------------------
def vec_norm(v: list[float]) -> float:
    """L2-норма вектора: sqrt(Σ x_i²). Это «длина» вектора."""
    return math.sqrt(sum(x * x for x in v))


def vec_dot(a: list[float], b: list[float]) -> float:
    """Скалярное произведение: Σ a_i · b_i."""
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Создаём приложение и поднимаем долгоживущие соединения один раз при старте.
# Это правильный паттерн: на каждый HTTP-запрос НЕ открываем заново сокет к
# Postgres и не пересоздаём httpx-клиент. Они потокобезопасны для нашего
# сценария.
# ---------------------------------------------------------------------------
app = FastAPI(title="RAG demo", version="1.0")

# CORS — фронт на :3000 ходит на :8000 напрямую (минует Next.js dev-proxy,
# который буферизует SSE). Для production-деплоя origins расширятся.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Эти объекты заполнятся в startup-хендлере ниже.
_embedder: LMStudioEmbedder | None = None
_store: VectorStore | None = None
_generator: ChatGenerator | None = None
_reranker: CrossEncoderReranker | None = None
_decomposer: QueryDecomposer | None = None
_rewriter: QueryRewriter | None = None
_history_store: ChatHistoryStore | None = None
_router: QueryRouter | None = None


@app.on_event("startup")
def _startup() -> None:
    """
    Открываем долгоживущие соединения при старте процесса.

    Reranker — отдельный кусок: при первой инициализации он скачает
    модель ~570 МБ с HuggingFace Hub. Дальше — мгновенный старт из кэша.
    Чтобы не блокировать запуск, если модель ещё не скачана — инициализация
    в try/except: если что-то пошло не так, сервер всё равно стартует,
    просто rerank-параметр будет отдавать 503.
    """
    global _embedder, _store, _generator, _reranker, _decomposer, _rewriter
    global _history_store, _router
    _embedder = LMStudioEmbedder()
    _store = VectorStore()
    _generator = ChatGenerator()
    # Decomposer и Rewriter переиспользуют HTTP-клиент генератора (тот же
    # endpoint /chat/completions). Стартуют мгновенно.
    _decomposer = QueryDecomposer(_generator)
    _rewriter = QueryRewriter(_generator)
    # History store использует то же psycopg-соединение что и VectorStore.
    _history_store = ChatHistoryStore(_store._conn)
    # Router — тот же HTTP-клиент chat-модели, классифицирует запросы.
    _router = QueryRouter(_generator)
    try:
        _reranker = CrossEncoderReranker()
    except Exception as exc:
        # Не критично для остального API — просто rerank будет недоступен.
        print(f"⚠ Reranker не инициализирован: {exc}")
        _reranker = None


@app.on_event("shutdown")
def _shutdown() -> None:
    """Аккуратно закрываем всё при остановке сервера."""
    if _reranker is not None:
        _reranker.close()
    if _embedder is not None:
        _embedder.close()
    if _store is not None:
        _store.close()
    if _generator is not None:
        _generator.close()


# ---------------------------------------------------------------------------
# Pydantic-схемы запроса/ответа. Pydantic автоматически валидирует входные
# JSON-ы и формирует OpenAPI-схему (доступна на /docs).
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Текст вопроса")
    top_k: int = Field(default=settings.top_k, ge=1, le=20)
    # Порог similarity для top-1. Если top-1.similarity < min_similarity,
    # в ответе ставим флаг below_threshold=True. LLM при этом всё равно
    # вызывается — мы просто помечаем что контекст ненадёжен.
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    # Режим поиска: vector | text | hybrid. По умолчанию hybrid — он
    # обычно даёт лучший recall, особенно на коротких запросах с точными
    # терминами.
    search_mode: str = Field(default="hybrid")
    # Включить ли LLM-based reranking поверх retrieve-этапа. По умолчанию
    # выключен — он медленный (несколько секунд на запрос).
    rerank: bool = Field(default=False)
    # Включить ли LLM-декомпозицию: вопрос → массив атомарных подзапросов.
    # Для каждого подзапроса будет отдельный retrieve, кандидаты сольются
    # через RRF. Полезно когда в одном запросе несколько разных тем.
    decompose: bool = Field(default=False)
    # Per-subquery rerank: rerank делается ОТДЕЛЬНО для каждого подзапроса,
    # потом топы объединяются. Даёт сбалансированный top-K где представлены
    # все темы запроса. Требует decompose=True (иначе игнорируется).
    rerank_per_subquery: bool = Field(default=False)
    # MMR (Maximal Marginal Relevance) — финальный шаг отбора, балансирует
    # релевантность и разнообразие. Помогает когда top_k забивается
    # дубликатами/соседними чанками.
    mmr: bool = Field(default=False)
    # Параметр λ для MMR ∈ [0, 1]. 1.0 = чистая релевантность,
    # 0.0 = чистое разнообразие, 0.5 — баланс.
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    # Порог reranker_score: чанки ниже этого значения отрезаются из финала.
    # 0.0 = фильтр выключен. Применяется только если rerank=True (нужны
    # reranker_score у чанков). Полезен на сложных запросах где система
    # вынужденно подбирает «шум» под подзапросы без релевантного материала.
    min_rerank_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Расширение контекста: к каждому финальному чанку для промпта LLM
    # подмешиваем соседей по chunk_index. Помогает когда смысл разорван
    # на границе чанков. В UI центральные чанки остаются прежними.
    expand_context: bool = Field(default=False)
    # Радиус расширения: 1 = по одному соседу с каждой стороны (±1),
    # 2 = ±2, и т.д. Большие радиусы быстро раздувают промпт.
    expand_radius: int = Field(default=1, ge=0, le=3)
    # Query rewriting: LLM генерирует N формулировок одного вопроса
    # (синонимы / технические термины / разная развёрнутость). Каждая
    # отдельно идёт в hybrid_search, кандидаты сливаются через RRF.
    # Расширяет «зону покрытия» в семантическом пространстве.
    # Игнорируется если decompose=True (взаимоисключающие режимы).
    rewrite: bool = Field(default=False)
    # Сколько формулировок генерировать (включая исходную). 1 = no-op.
    rewrite_n: int = Field(default=3, ge=1, le=5)
    # Conversation mode. Если передан — система:
    #   1) загружает последние сообщения чата и переформулирует текущий
    #      запрос в standalone-вид (резолвит «это», «он», «расскажи подробнее»);
    #   2) при retrieve видит общую базу + приватные чанки этого чата;
    #   3) сохраняет user-вопрос и assistant-ответ в messages.
    chat_id: str | None = None
    # Auto-router: классифицировать запрос ПЕРЕД retrieve. Если intent не
    # "knowledge" — отвечаем напрямую без RAG. Экономит время на chitchat
    # и meta-вопросах.
    auto_route: bool = False
    # Принудительный bypass RAG: пользователь явно выбрал режим без
    # retrieval. В отличие от auto_route, тут не запускается классификатор —
    # сразу идём в _generate_direct_answer с intent="general".
    bypass_rag: bool = False


class ChunkOut(BaseModel):
    source: str
    chunk_index: int
    content: str
    similarity: float
    # Метаданные ранжирования. None если чанк не попал в соответствующий рейтинг.
    vector_rank: int | None = None
    text_rank: int | None = None
    text_score: float | None = None
    rrf_score: float | None = None
    # Reranker-метрики (заполнены если rerank=True)
    reranker_score: float | None = None
    original_rank: int | None = None
    # MMR-метрика — позиция чанка после диверсификации.
    mmr_rank: int | None = None


class ScoredOut(BaseModel):
    """Лёгкая запись для гистограммы по всем чанкам."""
    source: str
    chunk_index: int
    similarity: float


class ExplainOut(BaseModel):
    """
    Образовательные подробности: как именно посчитан результат.
    Возвращается в meta чтобы UI мог нарисовать пошаговый расчёт.
    """
    # ── про эмбеддинг запроса ─────────────────────────────────────────
    embed_model: str
    embed_dim: int            # размерность вектора (для bge-m3 = 1024)
    embed_ms: float           # сколько мс заняла эмбеддинг-операция
    query_norm: float         # ||q|| — длина вектора запроса
    query_preview: list[float]  # первые ~32 значения для визуализации
    # ── про top-1 чанк (для пошагового cosine) ────────────────────────
    # None если поиск ничего не нашёл (база пуста).
    top_chunk_preview: list[float] | None = None
    top_chunk_norm: float | None = None
    top_dot: float | None = None        # q · d
    top_similarity: float | None = None  # = dot / (||q||·||d||)
    # ── распределение similarity по ВСЕМ чанкам базы ──────────────────
    all_scores: list[ScoredOut]
    # ── порог релевантности (для UI-предупреждения) ───────────────────
    # Сравниваем именно top-1: если он ниже — весь контекст слабый.
    min_similarity: float
    below_threshold: bool
    # ── режим поиска (для UI чтобы знать как рисовать чанки) ──────────
    search_mode: str
    # ── reranking-сводка ─────────────────────────────────────────────
    reranked: bool
    rerank_ms: float | None = None  # сколько времени заняло (в мс), если делалось
    # Режим reranking'а: "off" | "global" | "per_subquery".
    # "global"        — один rerank по исходному запросу (классика).
    # "per_subquery"  — rerank на каждом подзапросе отдельно, потом объединение.
    rerank_mode: str = "off"
    # ── MMR-сводка ─────────────────────────────────────────────────
    mmr_applied: bool = False
    mmr_lambda: float | None = None
    mmr_candidates: int | None = None  # сколько чанков было до MMR-обрезки
    # ── threshold-фильтр по reranker_score ─────────────────────────
    min_rerank_score: float = 0.0
    # Сколько чанков отрезано фильтром (информация для UI/предупреждения).
    filtered_by_threshold: int = 0
    # ── расширение контекста соседями ──────────────────────────────
    context_expanded: bool = False
    expand_radius: int | None = None
    neighbors_added: int = 0  # сколько уникальных соседних чанков подмешано
    # ── conversation mode (history-aware standalone rewrite) ──────
    chat_id: str | None = None
    standalone_query: str | None = None  # переписанный запрос (если был history)
    standalone_changed: bool = False
    standalone_ms: float | None = None
    history_used: int = 0  # сколько сообщений учтено
    # ── auto-router (классификатор запроса) ───────────────────────
    routed: bool = False           # делалась ли классификация
    route_intent: str | None = None  # "knowledge" | "chitchat" | "meta" | "other"
    route_reason: str | None = None
    route_ms: float | None = None
    route_fallback: bool = False   # был ли fallback на knowledge
    # Если интент не knowledge — RAG-этапы пропущены.
    rag_skipped: bool = False
    # Auto-fallback: router сказал knowledge, retrieve отработал, но top-1
    # similarity ниже порога — вместо «в контексте нет информации» отвечаем
    # из общей LLM-эрудиции. retrieve-данные (chunks/all_scores) остаются
    # в explain для прозрачности в drawer'е.
    auto_fallback: bool = False
    # ── query rewriting ────────────────────────────────────────────
    rewritten: bool = False             # реально ли сгенерировали >1 формулировку
    rewrite_status: str = "off"         # "off" | "rewritten" | "failed"
    rewrites: list[str] = []            # все формулировки (flat, для retrieval)
    rewrite_ms: float | None = None     # суммарное время LLM-вызовов
    # Группировка: если был decompose, для каждого подвопроса свои rewrites.
    # Каждая запись: {"subquery": str, "rewrites": list[str]}.
    # Если decompose выключен — одна группа c subquery=исходный_запрос.
    rewrite_groups: list[dict] = []
    # ── декомпозиция ─────────────────────────────────────────────────
    decomposed: bool                  # делался ли реальный разбор (status="decomposed")
    decompose_status: str = "off"     # "off" | "decomposed" | "atomic" | "failed"
    subqueries: list[str] = []        # что вернул декомпозитор (минимум — [query])
    decompose_ms: float | None = None # время на LLM-вызов


class AskResponse(BaseModel):
    chunks: list[ChunkOut]
    prompt: str  # итоговый user-промпт (отдаём чтобы показать в UI)
    answer: str
    explain: ExplainOut


def _to_chunk_out(chunks: list[RetrievedChunk]) -> list[ChunkOut]:
    """Конвертер dataclass → pydantic для отдачи в JSON."""
    return [
        ChunkOut(
            source=c.source,
            chunk_index=c.chunk_index,
            content=c.content,
            similarity=c.similarity,
            vector_rank=c.vector_rank,
            text_rank=c.text_rank,
            text_score=c.text_score,
            rrf_score=c.rrf_score,
            reranker_score=c.reranker_score,
            original_rank=c.original_rank,
            mmr_rank=c.mmr_rank,
        )
        for c in chunks
    ]


# Сколько первых значений вектора показывать в UI. Полные 1024 не нужны:
# для визуализации достаточно ~64 — глаз и так не различит больше.
PREVIEW_DIMS = 64


def _multi_query_per_subq_rerank(
    subqueries: list[str],
    top_k: int,
    candidate_per_subq: int = 15,
    chat_id: str | None = None,
) -> list[RetrievedChunk]:
    """
    Per-subquery rerank: для КАЖДОГО подзапроса делаем
        hybrid_search → rerank по ЭТОМУ подзапросу → top-N_per_subq
    Потом объединяем все top'ы, дедуплицируем по (source, chunk_index),
    обрезаем до top_k.

    Главное отличие от обычного rerank:
      - Reranker считает score по СВОЕМУ подвопросу, не по общему запросу.
      - Поэтому чанк который отвечает только на одну часть может получить
        высокий score (он действительно отвечает на свой подвопрос) и
        попасть в финал. В «глобальном» rerank такой чанк проигрывал бы
        более «универсальным» кандидатам.

    Стратегия объединения: первая встреча чанка с лучшим rerank-score
    выигрывает. Если чанк попал в top'ы нескольких подзапросов, его
    показываем один раз с самым высоким score.
    """
    assert _embedder is not None and _store is not None and _reranker is not None
    from dataclasses import replace

    # Сколько брать от каждого подвопроса. Гарантируем минимум 2,
    # чтобы при 5 подзапросах и top_k=5 не получить 5 чанков по 1 от каждого
    # без запаса на дедуп.
    per_subq_take = max(2, top_k // max(1, len(subqueries)) + 1)

    best_by_key: dict[tuple[str, int], RetrievedChunk] = {}

    for subq in subqueries:
        subq_vec = _embedder.embed_one(subq)
        cand = _store.hybrid_search(
            subq, subq_vec,
            top_k=candidate_per_subq,
            candidate_n=20,
            chat_id=chat_id,
        )
        # Reranker по ЭТОМУ подзапросу.
        ranked = _reranker.rerank(subq, cand, top_k=per_subq_take)

        for c in ranked:
            key = (c.source, c.chunk_index)
            existing = best_by_key.get(key)
            # Keep the best score across subqueries. reranker_score
            # сопоставим между подзапросами (sigmoid у нашего bge-reranker).
            if existing is None or (
                (c.reranker_score or 0.0) > (existing.reranker_score or 0.0)
            ):
                best_by_key[key] = c

    # Сортируем итоговый набор по reranker_score, обрезаем до top_k.
    result = list(best_by_key.values())
    result.sort(key=lambda c: c.reranker_score or 0.0, reverse=True)
    return result[:top_k]


def _multi_query_hybrid(
    subqueries: list[str],
    top_k: int,
    candidate_per_subq: int,
    chat_id: str | None = None,
) -> list[RetrievedChunk]:
    """
    Запускает hybrid_search для каждого подзапроса и сливает результаты
    через RRF по ключу (source, chunk_index).

    Идея та же что в одиночном hybrid'е — комбинируем по рангам, без
    масштабирования скоров. Только теперь списков не 2 (vector + text),
    а 2 × N (по два списка от каждого подзапроса, но через hybrid_search
    они уже склеены).

    candidate_per_subq — сколько кандидатов брать у каждого подзапроса.
    Должно быть больше top_k, чтобы хорошие кандидаты из разных подзапросов
    встречались в финале.
    """
    assert _embedder is not None and _store is not None
    from collections import defaultdict
    from dataclasses import replace

    by_key: dict[tuple[str, int], RetrievedChunk] = {}
    rrf_scores: dict[tuple[str, int], float] = defaultdict(float)

    for subq in subqueries:
        subq_vec = _embedder.embed_one(subq)
        hits = _store.hybrid_search(
            subq, subq_vec,
            top_k=candidate_per_subq,
            candidate_n=20,
            chat_id=chat_id,
        )
        for rank, hit in enumerate(hits, start=1):
            key = (hit.source, hit.chunk_index)
            # Кладём первый встретившийся вариант чанка (с его метаданными
            # rank'ов от первого подзапроса где он попал). RRF выше всё
            # равно сложит вклады правильно.
            if key not in by_key:
                by_key[key] = hit
            rrf_scores[key] += 1.0 / (RRF_K + rank)

    # Прописываем итоговый rrf_score, сортируем, обрезаем до top_k.
    result: list[RetrievedChunk] = []
    for key, hit in by_key.items():
        result.append(replace(hit, rrf_score=rrf_scores[key]))
    result.sort(key=lambda c: c.rrf_score or 0.0, reverse=True)
    return result[:top_k]


def _expand_chunks_with_neighbors(
    chunks: list[RetrievedChunk],
    radius: int,
) -> tuple[list[RetrievedChunk], int]:
    """
    Возвращает (расширенные_чанки_для_промпта, сколько_соседей_добавлено).

    Идея: оригинальные «центральные» чанки оставляем как есть в UI и для
    ранжирования, но для текста промпта в LLM каждому чанку дописываем
    соседей до радиуса. Получается связный кусок документа вокруг попавшего
    в финал фрагмента.

    Соседи помечаются текстовыми разделителями `[…контекст до…]` и
    `[…контекст после…]` — LLM понимает что это окружение, а не сам
    центральный фрагмент.

    Дубли уникализируем по (source, chunk_index): если у двух центральных
    чанков общий сосед, он войдёт только в один (более ранний). Сам центр
    тоже не дублируется в собственном «контексте» соседа.
    """
    assert _store is not None
    if radius <= 0 or not chunks:
        return chunks, 0

    centers = [(c.source, c.chunk_index) for c in chunks]
    pool = _store.get_neighbors(centers, radius=radius)

    # Считаем сколько НОВЫХ чанков мы притащили (соседей сверх центров).
    center_set = set(centers)
    neighbor_keys = set(pool.keys()) - center_set
    neighbors_added = len(neighbor_keys)

    # Чтобы один и тот же сосед не «прирос» дважды (если он сосед сразу
    # двум центральным), помечаем чанки которые уже использованы как
    # contextual neighbours.
    consumed: set[tuple[str, int]] = set()

    from dataclasses import replace
    enriched: list[RetrievedChunk] = []
    for c in chunks:
        parts: list[str] = []
        # Префиксные соседи (chunk_index - radius .. chunk_index - 1).
        for d in range(radius, 0, -1):
            key = (c.source, c.chunk_index - d)
            if key in pool and key not in consumed and key not in center_set:
                parts.append(f"[…контекст до…]\n{pool[key]}")
                consumed.add(key)
        # Сам центр.
        parts.append(c.content)
        # Постфиксные соседи.
        for d in range(1, radius + 1):
            key = (c.source, c.chunk_index + d)
            if key in pool and key not in consumed and key not in center_set:
                parts.append(f"[…контекст после…]\n{pool[key]}")
                consumed.add(key)

        enriched_text = "\n\n".join(parts)
        enriched.append(replace(c, content=enriched_text))

    return enriched, neighbors_added


def _retrieve_with_explain(
    query: str,
    top_k: int,
    min_similarity: float = 0.0,
    search_mode: str = "hybrid",
    rerank: bool = False,
    decompose: bool = False,
    rerank_per_subquery: bool = False,
    mmr: bool = False,
    mmr_lambda: float = 0.5,
    min_rerank_score: float = 0.0,
    expand_context: bool = False,
    expand_radius: int = 1,
    rewrite: bool = False,
    rewrite_n: int = 3,
    chat_id: str | None = None,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk], ExplainOut]:
    """
    Делает весь retrieval-цикл и собирает образовательный отчёт.

    Возвращает (chunks_для_LLM, explain_для_UI).

    search_mode:
      - "vector" — чистый векторный поиск (как раньше).
      - "text"   — только полнотекстовый (BM25-подобный через ts_rank_cd).
                   Эмбеддинг запроса всё равно считаем — он нужен для
                   образовательных блоков и гистограммы.
      - "hybrid" — оба + RRF.
    """
    assert _embedder is not None and _store is not None
    if search_mode not in SEARCH_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"search_mode должен быть одним из {SEARCH_MODES}",
        )

    # 1. Эмбеддинг ИСХОДНОГО запроса — всегда. Используется в:
    #    - образовательной панели (heatmap, норма, превью);
    #    - гистограмме «similarity по всей базе» (она показывает близость
    #      базы к исходному вопросу, не к подзапросам);
    #    - single-query retrieval, если decompose выключен.
    t0 = time.perf_counter()
    query_vec = _embedder.embed_one(query)
    embed_ms = (time.perf_counter() - t0) * 1000.0

    # 1.5. Декомпозиция запроса (опционально). Если выключено —
    #      subqueries состоит из одного элемента (исходный query),
    #      и дальше всё работает как раньше.
    subqueries: list[str] = [query]
    decompose_ms: float | None = None
    decompose_status = "off"  # "off" | "decomposed" | "atomic" | "failed"
    if decompose:
        if _decomposer is None:
            raise HTTPException(503, "Decomposer не инициализирован")
        t_dec = time.perf_counter()
        result = _decomposer.decompose(query)
        subqueries = result.subqueries
        decompose_status = result.status
        decompose_ms = round((time.perf_counter() - t_dec) * 1000.0, 1)

    # 1.6. Query rewriting (опционально). Работает И поверх decompose:
    #      если декомпозитор разбил вопрос на subqueries, для КАЖДОГО
    #      делаем свой rewrite (N формулировок). Результат — flat-список
    #      ВСЕХ формулировок ВСЕХ подвопросов.
    #
    #      Зачем combo: decompose работает на уровне «тем» (разные сущности),
    #      rewrite — на уровне «лексики одной темы» (синонимы/термины).
    #      Вместе дают максимальное покрытие.
    rewrites: list[str] = [query]
    rewrite_groups: list[dict] = []
    rewrite_ms: float | None = None
    rewrite_status = "off"  # "off" | "rewritten" | "failed"
    rewrite_active = rewrite
    if rewrite_active:
        if _rewriter is None:
            raise HTTPException(503, "Rewriter не инициализирован")
        # База для rewrite: либо подвопросы (если decompose сработал),
        # либо исходный запрос (один элемент).
        base_queries = (
            subqueries if decompose_status == "decomposed" else [query]
        )
        all_rewrites: list[str] = []
        any_real_rewrite = False
        any_failed = False
        t_rw = time.perf_counter()
        for sq in base_queries:
            rw_result = _rewriter.rewrite(sq, n=rewrite_n)
            rewrite_groups.append({
                "subquery": sq,
                "rewrites": rw_result.rewrites,
                "status": rw_result.status,
            })
            all_rewrites.extend(rw_result.rewrites)
            if rw_result.status == "rewritten" and len(rw_result.rewrites) > 1:
                any_real_rewrite = True
            if rw_result.status == "failed":
                any_failed = True
        rewrite_ms = round((time.perf_counter() - t_rw) * 1000.0, 1)
        rewrites = all_rewrites
        if any_real_rewrite:
            rewrite_status = "rewritten"
        elif any_failed:
            rewrite_status = "failed"
        else:
            # Все группы вернули по одной формулировке — фактически no-op.
            rewrite_status = "rewritten"

    # 2. Поиск в выбранном режиме.
    #    Когда включён rerank, нам нужно МНОГО кандидатов (RERANK_CANDIDATE_N),
    #    чтобы reranker имел из чего выбирать. Иначе хороший чанк, который
    #    изначально оказался на 12-м месте, никогда не попадёт в top_k.
    #    Когда включён MMR, тоже нужен запас — иначе MMR-отбор не сможет
    #    «играть» с разнообразием.
    pool_size = max(MMR_POOL_SIZE if mmr else 0, top_k)
    fetch_n = max(RERANK_CANDIDATE_N if rerank else 0, pool_size)

    # Per-subquery rerank требует и decompose, и rerank, и реальной декомпозиции.
    # Если эти условия не выполнены — режим сам по себе деградирует в global rerank
    # (или совсем выключается).
    is_per_subq = (
        rerank_per_subquery
        and rerank
        and decompose
        and len(subqueries) > 1
    )

    # Multi-query путь: если LLM реально разложил запрос на несколько подзапросов,
    # принудительно используем hybrid-стратегию по каждому из них и сливаем
    # через RRF. Режимы vector/text по отдельности не используем — multi-query
    # уже включает оба слоя hybrid'а.
    rerank_ms_inline: float | None = None
    if is_per_subq:
        # Этот пайплайн САМ делает rerank внутри (на каждом подвопросе),
        # поэтому ниже глобальный rerank не нужен. Берём pool_size кандидатов
        # чтобы оставить запас для MMR (если он тоже включён).
        if _reranker is None:
            raise HTTPException(503, "Reranker не инициализирован")
        t_rr = time.perf_counter()
        chunks = _multi_query_per_subq_rerank(
            subqueries,
            top_k=pool_size,
            candidate_per_subq=RERANK_CANDIDATE_N,
            chat_id=chat_id,
        )
        rerank_ms_inline = round((time.perf_counter() - t_rr) * 1000.0, 1)
    elif rewrite_active and len(rewrites) > 1:
        # Rewrite приоритет над «голым» decompose: rewrites уже содержат
        # все формулировки всех подвопросов (если decompose=True), или
        # формулировки исходного запроса (если decompose=False).
        # Объединение через RRF.
        chunks = _multi_query_hybrid(
            rewrites,
            top_k=fetch_n,
            candidate_per_subq=max(fetch_n, 10),
            chat_id=chat_id,
        )
    elif decompose and len(subqueries) > 1:
        chunks = _multi_query_hybrid(
            subqueries,
            top_k=fetch_n,
            candidate_per_subq=max(fetch_n, 10),
            chat_id=chat_id,
        )
    elif search_mode == "vector":
        chunks = _store.search(
            query_vec, top_k=fetch_n, include_embeddings=True, chat_id=chat_id,
        )
    elif search_mode == "text":
        text_hits = _store.text_search(query, top_n=fetch_n, chat_id=chat_id)
        chunks = [
            RetrievedChunk(
                source=h.source,
                chunk_index=h.chunk_index,
                content=h.content,
                similarity=0.0,
                text_rank=rank,
                text_score=h.text_score,
            )
            for rank, h in enumerate(text_hits, start=1)
        ]
    else:  # hybrid
        chunks = _store.hybrid_search(
            query, query_vec, top_k=fetch_n, candidate_n=20, chat_id=chat_id,
        )

    # 2.5. Reranking — опционально. Три случая:
    #   (а) is_per_subq — rerank уже сделан внутри _multi_query_per_subq_rerank,
    #       просто фиксируем mode и ms.
    #   (б) rerank=True (но не per-subq) — классический «глобальный» rerank
    #       по исходному запросу поверх объединённых кандидатов.
    #   (в) rerank=False — обрезаем результат до top_k и идём дальше.
    rerank_ms: float | None = None
    actually_reranked = False
    rerank_mode = "off"

    if is_per_subq:
        rerank_ms = rerank_ms_inline
        actually_reranked = True
        rerank_mode = "per_subquery"
    elif rerank and chunks:
        if _reranker is None:
            raise HTTPException(
                status_code=503,
                detail="Reranker не инициализирован (модель ещё не скачана?)",
            )
        t_rr = time.perf_counter()
        # Если MMR включён — reranker оставляет pool_size кандидатов,
        # финальную обрезку до top_k делает MMR.
        chunks = _reranker.rerank(query, chunks, top_k=pool_size)
        rerank_ms = round((time.perf_counter() - t_rr) * 1000.0, 1)
        actually_reranked = True
        rerank_mode = "global"
    elif not rerank:
        # rerank выключен — обрезаем до pool_size (= top_k если MMR off,
        # иначе MMR_POOL_SIZE). Финальная обрезка до top_k — ниже.
        chunks = chunks[:pool_size]

    # 2.6. MMR — Maximal Marginal Relevance. Финальный отбор с балансом
    #      «релевантность ↔ разнообразие». Применяется ПОСЛЕ всех остальных
    #      слоёв (hybrid, rerank, per-subq). Берёт текущий pool кандидатов
    #      и обрезает до top_k с учётом diversity.
    mmr_applied = False
    mmr_candidates_n: int | None = None
    if mmr and len(chunks) > 1:
        # Подгружаем эмбеддинги всех кандидатов одним SQL — нужны для
        # попарных cosine similarity в MMR-формуле.
        keys = [(c.source, c.chunk_index) for c in chunks]
        embeddings_map = _store.get_embeddings(keys)
        mmr_candidates_n = len(chunks)
        chunks = mmr_select(
            chunks,
            embeddings=embeddings_map,
            top_k=top_k,
            lambda_=mmr_lambda,
        )
        mmr_applied = True
    else:
        # MMR не применяем — но всё равно обрезаем до top_k
        # (выше могли взять pool_size > top_k для запаса).
        chunks = chunks[:top_k]

    # 2.7. Threshold-фильтр по reranker_score.
    #      Применяется ТОЛЬКО если был rerank — иначе нечего фильтровать
    #      (reranker_score у чанков отсутствует). Отрезает чанки с
    #      reranker_score ниже порога — это «явный шум», подобранный
    #      под подзапросы без релевантного материала.
    filtered_count = 0
    if actually_reranked and min_rerank_score > 0.0:
        before = len(chunks)
        chunks = [
            c for c in chunks
            if (c.reranker_score or 0.0) >= min_rerank_score
        ]
        filtered_count = before - len(chunks)

    # 2.8. Context expansion: к каждому центральному чанку для промпта
    #      LLM подмешиваем соседей. UI продолжает видеть «центры», а
    #      сгенерированный prompt использует расширенные тексты.
    chunks_for_prompt = chunks
    neighbors_added = 0
    context_expanded = False
    if expand_context and chunks and expand_radius > 0:
        chunks_for_prompt, neighbors_added = _expand_chunks_with_neighbors(
            chunks, radius=expand_radius,
        )
        context_expanded = neighbors_added > 0

    # 3. Similarity для ВСЕХ чанков базы — для гистограммы в UI.
    #    Не зависит от search_mode: гистограмма всегда показывает cosine,
    #    это «карта семантической близости» базы к запросу.
    all_scores = _store.score_all(query_vec, chat_id=chat_id)

    # 4. Образовательные числа для top-1.
    q_norm = vec_norm(query_vec)
    top_preview = top_norm = top_dot = top_sim = None
    top_chunk_embedding: list[float] | None = None

    if chunks:
        first = chunks[0]
        # Если у top-1 ещё нет embedding (text/hybrid режим) — достаём.
        if first.embedding is not None:
            top_chunk_embedding = first.embedding
        else:
            top_chunk_embedding = _store.get_embedding(
                first.source, first.chunk_index
            )

    if top_chunk_embedding is not None:
        top_norm = vec_norm(top_chunk_embedding)
        top_dot = vec_dot(query_vec, top_chunk_embedding)
        top_sim = top_dot / (q_norm * top_norm) if q_norm and top_norm else 0.0
        top_preview = top_chunk_embedding[:PREVIEW_DIMS]

    # Для below_threshold ориентируемся на cosine similarity (top_sim),
    # а не на chunks[0].similarity. В text-режиме у chunks[0].similarity = 0,
    # но семантически чанк может быть отличным — порог проверяем по cosine.
    below = top_sim is not None and top_sim < min_similarity

    explain = ExplainOut(
        embed_model=settings.voyage_embedding_model,
        embed_dim=len(query_vec),
        embed_ms=round(embed_ms, 1),
        query_norm=q_norm,
        query_preview=query_vec[:PREVIEW_DIMS],
        top_chunk_preview=top_preview,
        top_chunk_norm=top_norm,
        top_dot=top_dot,
        top_similarity=top_sim,
        all_scores=[
            ScoredOut(
                source=s.source,
                chunk_index=s.chunk_index,
                similarity=s.similarity,
            )
            for s in all_scores
        ],
        min_similarity=min_similarity,
        below_threshold=below,
        search_mode=search_mode,
        reranked=actually_reranked,
        rerank_ms=rerank_ms,
        rerank_mode=rerank_mode,
        decomposed=decompose_status == "decomposed",
        decompose_status=decompose_status,
        subqueries=subqueries,
        decompose_ms=decompose_ms,
        mmr_applied=mmr_applied,
        mmr_lambda=mmr_lambda if mmr_applied else None,
        mmr_candidates=mmr_candidates_n,
        min_rerank_score=min_rerank_score,
        filtered_by_threshold=filtered_count,
        context_expanded=context_expanded,
        expand_radius=expand_radius if expand_context else None,
        neighbors_added=neighbors_added,
        rewritten=rewrite_status == "rewritten" and len(rewrites) > 1,
        rewrite_status=rewrite_status,
        rewrites=rewrites,
        rewrite_ms=rewrite_ms,
        rewrite_groups=rewrite_groups,
    )
    return chunks, chunks_for_prompt, explain


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "chunks_in_db": _store.count() if _store else 0}


_CHITCHAT_SYSTEM = (
    "Ты — дружелюбный ассистент. Отвечай коротко и тепло на простые социальные "
    "реплики (приветствия, благодарности, прощания). Не выдумывай ничего "
    "лишнего, не задавай встречных вопросов про базу знаний."
)

_META_SYSTEM = (
    "Ты — ассистент, который отвечает на вопросы пользователя про сам диалог. "
    "Используй переданную историю чата чтобы ответить, что говорил пользователь "
    "ранее, что ты ему отвечал, повторить или объяснить проще предыдущий ответ. "
    "Если в истории нет нужной информации — честно скажи об этом."
)

_OTHER_SYSTEM = (
    "Ты — ассистент локальной базы знаний по технической документации. "
    "Пользователь задал вопрос вне темы базы (например про погоду, новости, "
    "личные дела). Вежливо объясни что не можешь ответить на этот вопрос — "
    "у тебя есть только техническая документация. Не выдумывай ответ."
)

_GENERAL_SYSTEM = (
    "Ты — полезный ассистент. Отвечай по делу и кратко. Если пользователь "
    "просит помочь с задачей, для которой пригодилась бы база знаний — "
    "напомни, что в чате есть переключатель режима RAG и его можно "
    "включить или поставить в Auto."
)


def _generate_direct_answer(
    query: str,
    history: list[Message] | None,
    intent: str,
    stream: bool = False,
):
    """
    Прямой ответ LLM без RAG-контекста. system-prompt подбирается под intent.
    Для intent=meta история диалога подаётся как контекст в user-message.

    Если stream=True — возвращает итератор кусков; иначе строку.
    """
    assert _generator is not None
    if intent == "chitchat":
        system = _CHITCHAT_SYSTEM
    elif intent == "meta":
        system = _META_SYSTEM
    elif intent == "other":
        system = _OTHER_SYSTEM
    elif intent == "general":
        system = _GENERAL_SYSTEM
    else:
        # Шафтовая защита: пришёл не-поддерживаемый intent — отвечаем нейтрально.
        system = _CHITCHAT_SYSTEM

    # Для meta даём явную историю в user-промпте, иначе LLM не сможет
    # сослаться на «предыдущие сообщения». Для general — тоже подмешиваем
    # историю: пользователь явно выбрал режим без RAG, но LLM лучше
    # отвечает с памятью предыдущих реплик.
    if intent in ("meta", "general") and history:
        hist_block = "\n".join(
            f"{m.role}: {m.content}" for m in history
        )
        user_content = (
            f"История диалога:\n{hist_block}\n\n"
            f"Текущий вопрос пользователя: {query}"
        )
    else:
        user_content = query

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    payload = {
        "model": _generator._model,
        "stream": stream,
        "temperature": 0.4,
        "max_tokens": 512,
        "messages": messages,
    }
    url = f"{_generator._base_url}/chat/completions"

    if not stream:
        response = _generator._client.post(url, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] or ""

    # Streaming-режим — повторяем логику generate_stream.
    def gen():
        with _generator._client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    payload_chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = payload_chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece
    return gen()


def _resolve_standalone(
    query: str, chat_id: str | None
) -> tuple[str, dict]:
    """
    Если есть chat_id и в его истории что-то есть — переформулируем
    текущий запрос в standalone-вид через LLM. Возвращаем (новый_запрос,
    info_для_explain).
    """
    if not chat_id or _history_store is None or _generator is None:
        return query, {
            "standalone_query": None,
            "standalone_changed": False,
            "standalone_ms": None,
            "history_used": 0,
        }
    history = _history_store.get_recent(chat_id, limit=6)
    if not history:
        return query, {
            "standalone_query": None,
            "standalone_changed": False,
            "standalone_ms": None,
            "history_used": 0,
        }
    t0 = time.perf_counter()
    res = make_standalone_query(_generator, history, query)
    standalone_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return res.standalone, {
        "standalone_query": res.standalone,
        "standalone_changed": res.changed,
        "standalone_ms": standalone_ms,
        "history_used": len(history),
    }


def _route_query(
    query: str, chat_id: str | None,
) -> tuple[RouteDecision | None, dict]:
    """
    Запускает классификатор запроса. Возвращает (decision, info_для_explain).
    Если auto_route выключен снаружи — эту функцию вообще не зовут.
    """
    if _router is None:
        return None, {}
    history = []
    if chat_id and _history_store is not None:
        history = _history_store.get_recent(chat_id, limit=4)

    t0 = time.perf_counter()
    decision = _router.classify(query, history=history)
    route_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return decision, {
        "routed": True,
        "route_intent": decision.intent,
        "route_reason": decision.reason,
        "route_ms": route_ms,
        "route_fallback": decision.fallback,
    }


def _save_chat_messages(
    chat_id: str | None,
    user_q: str,
    assistant_a: str,
    *,
    chunks: list | None = None,
    explain: dict | None = None,
    prompt: str | None = None,
) -> None:
    """Сохраняем пару сообщений в историю чата (если chat_id передан).
    Для assistant-сообщения дополнительно пишем pipeline-снимок —
    UI потом перерисует «шаги думания» после перезагрузки."""
    if not chat_id or _history_store is None:
        return
    _history_store.save_message(chat_id, "user", user_q)
    _history_store.save_message(
        chat_id,
        "assistant",
        assistant_a,
        chunks=chunks,
        explain=explain,
        prompt=prompt,
    )


# Порог уверенности reranker'а (sigmoid 0..1): ниже него считаем, что
# модель НЕ нашла релевантного контекста и в Auto-режиме переключаемся
# на общую LLM. Эмпирически на bge-reranker-v2-m3: релевантные чанки
# дают 0.5-0.95, нерелевантные — почти 0. Запас выбран небольшой.
AUTO_FALLBACK_RERANK_THRESHOLD = 0.1


def _should_auto_fallback(
    *,
    auto_route: bool,
    route_intent: str | None,
    chunks: list[RetrievedChunk],
    explain: ExplainOut,
    rerank_on: bool,
    min_similarity: float,
) -> bool:
    """
    Решаем: в режиме Auto после retrieve — отвечать через RAG (контекст
    есть) или фоллбэкнуться на общую LLM (контекста нет).

    Сигнал зависит от того, был ли reranker:
      - rerank ON  — смотрим максимум reranker_score (откалиброван 0..1).
      - rerank OFF — смотрим максимум cosine по ВСЕЙ базе (all_scores).
    Это надёжнее, чем top_similarity из chunks[0]: после rerank там может
    оказаться чанк с низкой cosine (reranker переставил порядок).
    """
    if not auto_route or route_intent != "knowledge":
        return False
    if not chunks:
        return True

    if rerank_on:
        max_rerank = max(
            (c.reranker_score for c in chunks if c.reranker_score is not None),
            default=0.0,
        )
        return max_rerank < AUTO_FALLBACK_RERANK_THRESHOLD

    max_sim = max(
        (s.similarity for s in explain.all_scores), default=0.0,
    )
    return max_sim < min_similarity


def _empty_explain(**overrides) -> ExplainOut:
    """
    Минимальная заглушка для direct-answer-режима, чтобы UI не падал.
    Поля retrieve-этапа остаются дефолтными (нулевыми), conversation-поля
    можно дозаполнить через overrides.
    """
    assert _embedder is not None
    return ExplainOut(
        embed_model=settings.voyage_embedding_model,
        embed_dim=settings.embedding_dim,
        embed_ms=0.0,
        query_norm=0.0,
        query_preview=[],
        all_scores=[],
        min_similarity=0.0,
        below_threshold=False,
        search_mode="skipped",
        reranked=False,
        decomposed=False,
        rag_skipped=True,
        **overrides,
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Обычный (нестриминговый) RAG-запрос."""
    assert _generator is not None  # для тайп-чекера

    # 0. Bypass RAG: пользователь явно выбрал режим без retrieve.
    if req.bypass_rag:
        history = (
            _history_store.get_recent(req.chat_id, limit=8)
            if req.chat_id and _history_store else []
        )
        answer = _generate_direct_answer(
            req.query, history, intent="general", stream=False,
        )
        explain = _empty_explain(chat_id=req.chat_id)
        _save_chat_messages(
            req.chat_id, req.query, answer,
            explain=explain.model_dump(),
        )
        return AskResponse(
            chunks=[], prompt="", answer=answer, explain=explain,
        )

    # 0a. Router: классифицируем intent. Если включён и intent != knowledge —
    #     пропускаем RAG, отвечаем напрямую.
    route_info: dict = {}
    if req.auto_route:
        decision, route_info = _route_query(req.query, req.chat_id)
        if decision is not None and decision.intent != "knowledge":
            # Прямой ответ без retrieve.
            history = (
                _history_store.get_recent(req.chat_id, limit=8)
                if req.chat_id and _history_store else []
            )
            answer = _generate_direct_answer(
                req.query, history, decision.intent, stream=False,
            )
            explain = _empty_explain(
                chat_id=req.chat_id, **route_info,
            )
            # Сохраняем без chunks/prompt — RAG не запускался,
            # но explain (с router-решением) полезен для UI.
            _save_chat_messages(
                req.chat_id, req.query, answer,
                explain=explain.model_dump(),
            )
            return AskResponse(
                chunks=[], prompt="", answer=answer, explain=explain,
            )

    # 0b. Conversation-mode препроцессинг: history-aware standalone rewrite.
    effective_query, standalone_info = _resolve_standalone(req.query, req.chat_id)

    chunks, chunks_for_prompt, explain = _retrieve_with_explain(
        effective_query, req.top_k, req.min_similarity, req.search_mode,
        req.rerank, req.decompose, req.rerank_per_subquery,
        req.mmr, req.mmr_lambda, req.min_rerank_score,
        req.expand_context, req.expand_radius,
        req.rewrite, req.rewrite_n,
        chat_id=req.chat_id,
    )
    # Дописываем conversation-поля в explain (Pydantic не даёт мутировать
    # frozen-объекты — создаём копию через model_copy с update).
    explain = explain.model_copy(update={
        "chat_id": req.chat_id,
        **standalone_info,
        **route_info,  # router-инфо (пусто если auto_route выключен)
    })
    # 0c. Auto-fallback: router отправил в RAG (intent=knowledge), но ни
    #     один чанк не получил достаточной уверенности. Сигнал зависит от
    #     того, был ли rerank:
    #       - rerank ON  → top.reranker_score (sigmoid-калиброван, 0..1);
    #                       это правильный показатель «нашли ли релевантное».
    #       - rerank OFF → max cosine по ВСЕЙ базе (all_scores); top из
    #                       chunks ненадёжен (могут быть text-only).
    #     На bge-m3 для русского cosine релевантного чанка часто 0.35-0.5
    #     даже когда тема ровно в базе — поэтому полагаться только на cosine
    #     рискованно.
    should_fallback = _should_auto_fallback(
        auto_route=req.auto_route,
        route_intent=route_info.get("route_intent"),
        chunks=chunks,
        explain=explain,
        rerank_on=req.rerank,
        min_similarity=req.min_similarity,
    )
    if should_fallback:
        history = (
            _history_store.get_recent(req.chat_id, limit=8)
            if req.chat_id and _history_store else []
        )
        answer = _generate_direct_answer(
            req.query, history, intent="general", stream=False,
        )
        explain = explain.model_copy(update={
            "rag_skipped": True,
            "auto_fallback": True,
        })
        _save_chat_messages(
            req.chat_id, req.query, answer,
            chunks=[c.model_dump() for c in _to_chunk_out(chunks)],
            explain=explain.model_dump(),
        )
        return AskResponse(
            chunks=_to_chunk_out(chunks),
            prompt="",
            answer=answer,
            explain=explain,
        )

    # On-mode (или Auto при `chunks_in_db==0`): если запас чанков пуст —
    # это уже не fallback-кейс, отдаём явную ошибку про ingest.
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="В базе нет чанков. Запусти `python ingest.py`.",
        )

    # Промпт и ответ строим на расширенных чанках (с соседями, если был
    # expand_context). В UI отдаём оригинальные «центральные» чанки —
    # пользователь видит что именно retrieval нашёл, без раздувания.
    prompt = build_user_prompt(effective_query, chunks_for_prompt)
    answer = _generator.generate(effective_query, chunks_for_prompt)

    # Сохраняем оба сообщения в истории чата (если chat_id задан).
    # Пишем ИСХОДНЫЙ user-вопрос (а не standalone-переписанный) —
    # пользователь увидит свой текст в истории, а не машинный перепев.
    # Для assistant пишем полный pipeline-снимок (chunks/explain/prompt)
    # чтобы можно было показать «как я дошёл до ответа» после рестарта.
    _save_chat_messages(
        req.chat_id, req.query, answer,
        chunks=[c.model_dump() for c in _to_chunk_out(chunks)],
        explain=explain.model_dump(),
        prompt=prompt,
    )

    return AskResponse(
        chunks=_to_chunk_out(chunks),
        prompt=prompt,
        answer=answer,
        explain=explain,
    )


def _sse_event(event: str, data: dict | str) -> str:
    """
    Формирует одно SSE-сообщение.

    Формат Server-Sent Events:
        event: <имя>\n
        data: <payload>\n
        \n        ← пустая строка-разделитель обязательна

    Если в data многострочный текст, нужно писать "data: " перед каждой строкой,
    но мы шлём JSON-строку в одну строчку (через ensure_ascii=False), так что
    эта проблема нас не касается.
    """
    if isinstance(data, dict):
        data = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


@app.get("/api/ask/stream")
def ask_stream(
    query: str,
    top_k: int = settings.top_k,
    min_similarity: float = 0.0,
    search_mode: str = "hybrid",
    rerank: bool = False,
    decompose: bool = False,
    rerank_per_subquery: bool = False,
    mmr: bool = False,
    mmr_lambda: float = 0.5,
    min_rerank_score: float = 0.0,
    expand_context: bool = False,
    expand_radius: int = 1,
    rewrite: bool = False,
    rewrite_n: int = 3,
    chat_id: str | None = None,
    auto_route: bool = False,
    bypass_rag: bool = False,
) -> StreamingResponse:
    """
    Streaming-версия. Шлёт три типа событий:
      meta  — найденные чанки + промпт + образовательный explain
      token — очередной кусок текста (много раз по мере генерации)
      done  — конец (один раз)

    В браузере читается стандартным EventSource — см. index.html.
    """
    assert _generator is not None

    if not query.strip():
        raise HTTPException(status_code=400, detail="query пуст")

    # 0. Bypass RAG: пользователь явно выбрал режим без retrieve.
    if bypass_rag:
        history = (
            _history_store.get_recent(chat_id, limit=8)
            if chat_id and _history_store else []
        )
        explain = _empty_explain(chat_id=chat_id)

        def bypass_event_source() -> Iterator[str]:
            yield _sse_event("meta", {
                "chunks": [],
                "prompt": "",
                "explain": explain.model_dump(),
            })
            full: list[str] = []
            try:
                for piece in _generate_direct_answer(
                    query, history, intent="general", stream=True,
                ):
                    full.append(piece)
                    yield _sse_event("token", {"text": piece})
            finally:
                # Сохраняем даже при дисконнекте — иначе ответ уехал, но в БД его нет.
                if full:
                    _save_chat_messages(
                        chat_id, query, "".join(full),
                        explain=explain.model_dump(),
                    )
            yield _sse_event("done", {})

        return StreamingResponse(
            bypass_event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 0a. Router: классифицируем intent. Если включён и intent != knowledge —
    #     стримим прямой ответ без retrieve-этапа.
    route_info: dict = {}
    direct_intent: str | None = None
    if auto_route:
        decision, route_info = _route_query(query, chat_id)
        if decision is not None and decision.intent != "knowledge":
            direct_intent = decision.intent

    if direct_intent is not None:
        # Streaming-ветка для chitchat/meta/other.
        history = (
            _history_store.get_recent(chat_id, limit=8)
            if chat_id and _history_store else []
        )
        explain = _empty_explain(chat_id=chat_id, **route_info)

        def direct_event_source() -> Iterator[str]:
            yield _sse_event("meta", {
                "chunks": [],
                "prompt": "",
                "explain": explain.model_dump(),
            })
            full: list[str] = []
            try:
                for piece in _generate_direct_answer(
                    query, history, direct_intent, stream=True,
                ):
                    full.append(piece)
                    yield _sse_event("token", {"text": piece})
            finally:
                if full:
                    _save_chat_messages(
                        chat_id, query, "".join(full),
                        explain=explain.model_dump(),
                    )
            yield _sse_event("done", {})

        return StreamingResponse(
            direct_event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 0b. Conversation-mode препроцессинг: history-aware standalone rewrite.
    effective_query, standalone_info = _resolve_standalone(query, chat_id)

    chunks, chunks_for_prompt, explain = _retrieve_with_explain(
        effective_query, top_k, min_similarity, search_mode,
        rerank, decompose, rerank_per_subquery,
        mmr, mmr_lambda, min_rerank_score,
        expand_context, expand_radius,
        rewrite, rewrite_n,
        chat_id=chat_id,
    )
    explain = explain.model_copy(update={
        "chat_id": chat_id, **standalone_info, **route_info,
    })

    # 0c. Auto-fallback (см. POST /api/ask): отвечаем из общей LLM если
    #     уверенности retrieval'а не хватает.
    should_fallback = _should_auto_fallback(
        auto_route=auto_route,
        route_intent=route_info.get("route_intent"),
        chunks=chunks,
        explain=explain,
        rerank_on=rerank,
        min_similarity=min_similarity,
    )
    if should_fallback:
        explain = explain.model_copy(update={
            "rag_skipped": True,
            "auto_fallback": True,
        })

    def event_source() -> Iterator[str]:
        # 1. Сначала отдаём «meta»: что нашли, промпт и explain-блок.
        #    Используем _to_chunk_out (тот же, что в POST /api/ask) — иначе
        #    легко забыть какое-нибудь поле когда схема расширяется.
        #    В UI отдаём оригинальные центральные чанки, а в промпт —
        #    расширенные соседями (если включён expand_context).
        # Считаем промпт один раз — отдадим в meta И сохраним в БД для history.
        # При auto-fallback prompt пустой: ответ не использует контекст.
        prompt_text = (
            "" if should_fallback
            else build_user_prompt(effective_query, chunks_for_prompt)
        )
        chunks_serializable = [c.model_dump() for c in _to_chunk_out(chunks)]
        explain_dict = explain.model_dump()
        meta_payload = {
            "chunks": chunks_serializable,
            "prompt": prompt_text,
            "explain": explain_dict,
        }
        yield _sse_event("meta", meta_payload)

        # Если контекста нет — заканчиваем (только в On-режиме). В Auto
        # пустые chunks означают «reranker всё отрезал» → нам нужен
        # fallback на общую LLM, обработка ниже.
        if not chunks and not should_fallback:
            yield _sse_event(
                "token",
                {"text": "В базе нет чанков. Запусти `python ingest.py`."},
            )
            yield _sse_event("done", {})
            return

        # 2. Стримим токены ответа. При auto-fallback — из общей LLM-эрудиции;
        #    в обычном RAG-пути — с подмешанным контекстом.
        full_answer_parts: list[str] = []
        if should_fallback:
            history = (
                _history_store.get_recent(chat_id, limit=8)
                if chat_id and _history_store else []
            )
            token_iter = _generate_direct_answer(
                effective_query, history, intent="general", stream=True,
            )
        else:
            token_iter = _generator.generate_stream(
                effective_query, chunks_for_prompt,
            )
        try:
            for piece in token_iter:
                full_answer_parts.append(piece)
                yield _sse_event("token", {"text": piece})
        finally:
            # 3. Сохраняем сообщения в историю даже если клиент отвалился
            #    (GeneratorExit на yield во время стрима). Иначе ответ
            #    «уехал» в UI, но в БД его нет — после reload бабл пропадает.
            if full_answer_parts:
                full_answer = "".join(full_answer_parts)
                _save_chat_messages(
                    chat_id, query, full_answer,
                    chunks=chunks_serializable,
                    explain=explain_dict,
                    prompt=prompt_text if prompt_text else None,
                )

        # 4. Сигнал конца.
        yield _sse_event("done", {})

    # Важные заголовки для SSE:
    #   - text/event-stream — это и есть тип «потока событий»;
    #   - X-Accel-Buffering: no — отключает буферизацию в nginx/прокси,
    #     иначе токены будут копиться и идти большими порциями.
    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Chats и messages — conversation mode.
# ---------------------------------------------------------------------------
import uuid


class ChatOut(BaseModel):
    id: str
    title: str
    created_at: str | None = None


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: str | None = None
    # Опциональные данные о retrieval-pipeline (только для assistant-сообщений).
    # Позволяют UI перерендерить «шаги думания» после перезагрузки страницы.
    chunks: list | None = None
    explain: dict | None = None
    prompt: str | None = None


class CreateChatRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


def _generate_chat_title(query: str) -> str:
    """
    Простой эвристический заголовок чата из первого вопроса.
    Берём первые ~60 символов, схлопываем пробелы. Без LLM —
    это быстрее и стабильнее (не лезем в Gemma на каждый новый чат).
    """
    title = " ".join((query or "").split())[:60]
    return title or "Без названия"


@app.post("/api/chats", response_model=ChatOut)
def create_chat(req: CreateChatRequest) -> ChatOut:
    """Создаёт новый чат с заданным или дефолтным заголовком."""
    assert _store is not None
    chat_id = str(uuid.uuid4())
    title = (req.title or "Новый чат").strip() or "Новый чат"
    with _store._conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chats (id, title) VALUES (%s, %s) "
            "RETURNING created_at",
            (chat_id, title),
        )
        row = cur.fetchone()
    created_at = row[0].isoformat() if row and row[0] is not None else None
    return ChatOut(id=chat_id, title=title, created_at=created_at)


@app.get("/api/chats", response_model=list[ChatOut])
def list_chats() -> list[ChatOut]:
    """Все чаты, новые сверху."""
    assert _store is not None
    with _store._conn.cursor() as cur:
        cur.execute(
            "SELECT id, title, created_at FROM chats ORDER BY created_at DESC"
        )
        rows = cur.fetchall()
    return [
        ChatOut(
            id=r[0],
            title=r[1],
            created_at=r[2].isoformat() if r[2] is not None else None,
        )
        for r in rows
    ]


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str) -> dict:
    """
    Удаляет чат вместе с его сообщениями (CASCADE на FK) и его
    приватными чанками (chunks.chat_id = ...).
    """
    assert _store is not None
    with _store._conn.cursor() as cur:
        # 1) приватные чанки этого чата
        cur.execute("DELETE FROM chunks WHERE chat_id = %s", (chat_id,))
        chunks_deleted = cur.rowcount
        # 2) сам чат (messages снесутся каскадом)
        cur.execute("DELETE FROM chats WHERE id = %s", (chat_id,))
        chat_deleted = cur.rowcount
    return {
        "chat_id": chat_id,
        "deleted": chat_deleted > 0,
        "chunks_deleted": chunks_deleted,
    }


@app.get("/api/chats/{chat_id}/messages", response_model=list[MessageOut])
def list_messages(chat_id: str, limit: int = 100) -> list[MessageOut]:
    """Последние N сообщений чата в хронологическом порядке.
    Возвращаем с полными pipeline-данными — UI рендерит шаги для
    каждого assistant-сообщения, не только последнего."""
    assert _history_store is not None
    msgs = _history_store.get_recent(chat_id, limit=limit, with_details=True)
    return [
        MessageOut(
            role=m.role,
            content=m.content,
            created_at=m.created_at,
            chunks=m.chunks,
            explain=m.explain,
            prompt=m.prompt,
        )
        for m in msgs
    ]


@app.post("/api/admin/cleanup")
def cleanup_old_chats(days: int = 7) -> dict:
    """
    Удаляет чаты и их приватные данные старше N дней.

    Что чистится:
      - chats старше days дней (messages удаляются каскадно)
      - chunks с chat_id != NULL у удалённых чатов (chunks не имеют FK,
        поэтому удаляем отдельно)
      - «сиротские» приватные чанки чатов, которых уже нет

    Общая база (chunks.chat_id IS NULL) НЕ трогается никогда.

    В production обычно вызывают по cron каждый час. В нашем демо —
    ручной endpoint, удобно дёрнуть из curl при тестах.
    """
    assert _store is not None
    with _store._conn.cursor() as cur:
        # 1) сиротские приватные чанки (чат уже удалён, чанки висят)
        cur.execute(
            "DELETE FROM chunks WHERE chat_id IS NOT NULL AND chat_id NOT IN "
            "(SELECT id FROM chats)"
        )
        orphans = cur.rowcount
        # 2) старые чаты — каскад снесёт messages
        cur.execute(
            "DELETE FROM chats WHERE created_at < now() - "
            "(%s * interval '1 day')",
            (days,),
        )
        old_chats = cur.rowcount
        # 3) приватные чанки уже несуществующих чатов (после step 2)
        cur.execute(
            "DELETE FROM chunks WHERE chat_id IS NOT NULL AND chat_id NOT IN "
            "(SELECT id FROM chats)"
        )
        more_orphans = cur.rowcount
    return {
        "days_threshold": days,
        "orphan_chunks_before": orphans,
        "old_chats_deleted": old_chats,
        "chunks_deleted_after": more_orphans,
    }


# ---------------------------------------------------------------------------
# Загрузка файлов и управление источниками.
# ---------------------------------------------------------------------------
class SourceOut(BaseModel):
    """Описание источника в списке."""
    source: str
    chunks: int
    created_at: str | None = None
    chat_id: str | None = None


class UploadResult(BaseModel):
    source: str
    chunks: int


def _index_text(text: str, source: str, chat_id: str | None = None) -> int:
    """
    Общая логика «текст → чанки → эмбеддинги → INSERT».
    Идемпотентна: сначала удаляет старые чанки этого `source` (с тем же
    chat_id), потом вставляет новые. Возвращает количество чанков.
    """
    assert _embedder is not None and _store is not None
    chunks = chunk_text(
        text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    if not chunks:
        return 0

    # Idempotent re-ingest: удаляем именно ту версию которая загружается.
    # При chat_id=None — удаляем версию из общей базы; иначе только из
    # этого чата. Версии в других чатах не трогаем.
    _store.delete_source(source, chat_id=chat_id)

    # Эмбеддинги — пачками, чтобы не плодить HTTP-вызовы.
    BATCH = 16
    records: list[ChunkInput] = []
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start : start + BATCH]
        vectors = _embedder.embed_many([c.text for c in batch])
        for chunk, vec in zip(batch, vectors):
            records.append(ChunkInput(
                source=source,
                chunk_index=chunk.index,
                content=chunk.text,
                embedding=vec,
                chat_id=chat_id,
            ))
    _store.insert_chunks(records)
    return len(records)


@app.get("/api/sources", response_model=list[SourceOut])
def list_sources(chat_id: str | None = None) -> list[SourceOut]:
    """
    Возвращает список загруженных источников.
    Без chat_id — показывает общую базу (chat_id IS NULL).
    """
    assert _store is not None
    rows = _store.list_sources(chat_id=chat_id)
    return [SourceOut(**r) for r in rows]


@app.post("/api/upload", response_model=UploadResult)
async def upload_file(
    file: UploadFile = File(...),
    chat_id: str | None = None,
) -> UploadResult:
    """
    Принимает один файл (.md/.txt/.pdf), парсит его в текст, чанкует,
    эмбеддит и сохраняет в БД.

    Idempotent: повторная загрузка с тем же именем заменяет старые чанки.
    chat_id оставляем как опцию для будущего chat-режима — сейчас все
    загрузки идут в общую базу (chat_id=NULL).
    """
    if file.filename is None:
        raise HTTPException(400, "Не указано имя файла")

    data = await file.read()

    try:
        text = load_text(file.filename, data)
    except UnsupportedFile as exc:
        raise HTTPException(400, str(exc)) from exc

    # Имя источника — оригинальное имя файла. Не делаем cwd-санитации,
    # потому что в БД оно — просто текст, не путь.
    n_chunks = _index_text(text, source=file.filename, chat_id=chat_id)
    if n_chunks == 0:
        raise HTTPException(
            400,
            "Файл прочитан, но не дал чанков (возможно пустой текст)",
        )
    return UploadResult(source=file.filename, chunks=n_chunks)


@app.delete("/api/sources/{source}")
def delete_source(source: str) -> dict:
    """
    Удаляет все чанки указанного источника. Возвращает счётчик удалённого.
    """
    assert _store is not None
    deleted = _store.delete_source(source)
    return {"source": source, "deleted": deleted}


# ---------------------------------------------------------------------------
# Evaluation endpoint — прогон голден-сета через все конфиги.
# ---------------------------------------------------------------------------
def _run_evaluation(
    items: list[GoldenItem],
    top_k: int,
    include_rerank: bool,
) -> dict:
    """
    Прогоняет голден-сет через все retrieval-конфиги и собирает метрики.

    Делаем синхронно — на 30 запросах и без rerank это ~3 секунды, с rerank ~30.
    Для учебного демо нормально, в production выносят в background-task.
    """
    assert _embedder is not None and _store is not None

    configs: list[tuple[str, object]] = [
        ("vector", make_vector_retriever(_embedder, _store, top_k)),
        ("text",   make_text_retriever(_store, top_k)),
        ("hybrid", make_hybrid_retriever(_embedder, _store, top_k)),
    ]
    if include_rerank:
        if _reranker is None:
            raise HTTPException(
                status_code=503,
                detail="Reranker не инициализирован",
            )
        configs.append((
            "hybrid+rerank",
            make_hybrid_rerank_retriever(_embedder, _store, _reranker, top_k),
        ))

    aggregates: dict[str, dict] = {}
    details_by_config: dict[str, list[tuple[GoldenItem, QueryMetrics]]] = {}

    for name, retriever in configs:
        per_query: list[QueryMetrics] = []
        latencies: list[float] = []
        details: list[tuple[GoldenItem, QueryMetrics]] = []
        for item in items:
            t0 = time.perf_counter()
            retrieved = retriever(item.question)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            m = score_query(retrieved, item.relevant, k=top_k)
            per_query.append(m)
            details.append((item, m))

        agg = aggregate(per_query, latencies)
        aggregates[name] = {
            "n_queries": agg.n_queries,
            "hit_rate": agg.hit_rate,
            "recall_at_k": agg.recall_at_k,
            "precision_at_k": agg.precision_at_k,
            "mrr": agg.mrr,
            "avg_latency_ms": agg.avg_latency_ms,
        }
        details_by_config[name] = details

    # Разбивка по тегам: для каждого тега → MRR каждого конфига.
    all_tags: list[str] = []
    for item in items:
        for t in item.tags:
            if t not in all_tags:
                all_tags.append(t)

    by_tag: list[dict] = []
    for tag in all_tags:
        n = sum(1 for it in items if tag in it.tags)
        row: dict = {"tag": tag, "n": n}
        for cfg_name in aggregates:
            tag_metrics = split_by_tag(details_by_config[cfg_name], tag)
            if tag_metrics:
                row[cfg_name] = (
                    sum(m.reciprocal_rank for m in tag_metrics) / len(tag_metrics)
                )
            else:
                row[cfg_name] = None
        by_tag.append(row)

    return {
        "n_queries": len(items),
        "top_k": top_k,
        "aggregates": aggregates,
        "by_tag": by_tag,
    }


class EvalRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20)
    include_rerank: bool = Field(default=False)


@app.post("/api/eval")
def run_evaluation(req: EvalRequest) -> dict:
    """
    Прогон голден-сета. По умолчанию без rerank (тогда быстро, ~3 сек),
    с include_rerank=True долго (~30 сек) но видно как cross-encoder
    помогает на сложных запросах.
    """
    items = load_golden_set()
    return _run_evaluation(items, req.top_k, req.include_rerank)


# ---------------------------------------------------------------------------
# Раздача статики и корневая страница.
# ---------------------------------------------------------------------------
_STATIC_DIR = Path(__file__).parent / "static"

# /static/* отдаёт файлы напрямую (на случай если в будущем появятся
# отдельные css/js/картинки).
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    """Главная страница — наш HTML с интерфейсом.

    Cache-Control: no-cache важен в учебном проекте: мы часто правим HTML/JS,
    а браузер по умолчанию кеширует ответ — пользователь получает старую
    версию после правок до жёсткого reload'а. no-cache заставит браузер
    каждый раз сходить за свежей версией.
    """
    return FileResponse(
        _STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ============================================================================
# Agent (Spec 1: LangGraph-based skkrondo concierge)
# ============================================================================

class AgentAskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class TraceEvent(BaseModel):
    type: str
    timestamp: str
    data: dict


class AgentAskResponse(BaseModel):
    answer: str
    trace: list[TraceEvent]
    iterations: int


class AgentMessageOut(BaseModel):
    id: int
    role: str
    content: str
    trace: list[TraceEvent] | None = None
    created_at: str


def _save_agent_message(role: str, content: str, trace: list | None) -> int:
    """Сохраняет один agent_message в БД. Возвращает id вставленной строки."""
    assert _store is not None
    with _store._conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO agent_messages (role, content, trace)
            VALUES (%s, %s, %s::jsonb)
            RETURNING id
            """,
            (role, content, json.dumps(trace) if trace is not None else None),
        )
        row = cur.fetchone()
    return row[0]


def _list_agent_messages(limit: int = 200) -> list[AgentMessageOut]:
    assert _store is not None
    with _store._conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, role, content, trace, created_at
            FROM agent_messages
            ORDER BY id ASC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
    return [
        AgentMessageOut(
            id=r[0],
            role=r[1],
            content=r[2],
            trace=r[3] if r[3] is not None else None,
            created_at=r[4].isoformat(),
        )
        for r in rows
    ]


async def _save_agent_message_async(role: str, content: str, trace: list | None) -> int:
    """Async-обёртка над sync psycopg-вызовом. psycopg-коннект — синхронный
    (autocommit=True от VectorStore), а вызывается из async-эндпоинта —
    без to_thread INSERT блокирует event loop на время round-trip'а к Postgres."""
    return await asyncio.to_thread(_save_agent_message, role, content, trace)


# In-memory накопление trace по thread_id: clarify-круги и финальный ответ
# размазаны по нескольким HTTP-запросам одного turn'а. Парно с MemorySaver
# (тоже in-memory). На `done` сохраняем полный trace и чистим запись.
_agent_trace_buffers: dict[str, list[dict]] = {}

# Кап на число «живых» turn'ов (буферов trace), которые ждут подтверждения.
# Если пользователь бросает clarify не ответив, запись висит вечно — здесь
# вытесняем самые старые (dict хранит порядок вставки), чтобы не течь.
_MAX_LIVE_TURNS = 100

# Лимит длины уточнения пользователя на «нет» (как query в AgentAskRequest).
_MAX_CORRECTION_LEN = 2000


def _register_turn_buffer(thread_id: str) -> None:
    """Заводит буфер для нового turn'а, вытесняя самый старый при переполнении."""
    while len(_agent_trace_buffers) >= _MAX_LIVE_TURNS:
        oldest = next(iter(_agent_trace_buffers))
        _agent_trace_buffers.pop(oldest, None)
    _agent_trace_buffers[thread_id] = []


@app.post("/api/agent/ask", response_model=AgentAskResponse)
async def agent_ask(req: AgentAskRequest) -> AgentAskResponse:
    """Non-streaming прогон агента (авто-подтверждение). Сохраняет user+assistant."""
    thread_id = uuid.uuid4().hex
    await _save_agent_message_async("user", req.query, None)
    try:
        result = await run_collect(req.query, thread_id=thread_id)
    except Exception as exc:
        await _save_agent_message_async("assistant", f"Внутренняя ошибка: {exc}", [])
        raise
    await _save_agent_message_async("assistant", result.answer, result.trace)
    return AgentAskResponse(
        answer=result.answer,
        trace=[TraceEvent(**e) for e in result.trace],
        iterations=result.iterations,
    )


@app.get("/api/agent/messages", response_model=list[AgentMessageOut])
def agent_messages(limit: int = 200) -> list[AgentMessageOut]:
    return _list_agent_messages(limit=limit)


@app.get("/api/agent/ask/stream")
async def agent_ask_stream(query: str) -> StreamingResponse:
    """SSE-стрим старта turn'а. Генерит thread_id, стримит до первого clarify.
    user-сообщение сохраняется здесь; assistant — на событии done (в resume)."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="query пуст")

    thread_id = uuid.uuid4().hex
    await _save_agent_message_async("user", query, None)
    _register_turn_buffer(thread_id)

    async def event_source():
        async for event in _agent_turn_events(
            thread_id, query=query, resume=None,
        ):
            yield _sse_event(event["type"], event["data"])

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _agent_turn_events(thread_id: str, *, query, resume):
    """Прогон одного шага turn'а (старт или resume) с накоплением trace.
    На done сохраняет assistant-сообщение с полным trace и чистит буфер."""
    buf = _agent_trace_buffers.setdefault(thread_id, [])
    final_answer = ""
    async for event in run_stream(thread_id, query=query, resume=resume):
        buf.append(event)
        if event["type"] == "final_answer":
            final_answer = event["data"]["text"]
        if event["type"] in ("done", "error"):
            await _save_agent_message_async("assistant", final_answer, list(buf))
            _agent_trace_buffers.pop(thread_id, None)
        yield event


@app.get("/api/agent/resume/stream")
async def agent_resume_stream(
    thread_id: str,
    confirmed: bool,
    correction: str = "",
) -> StreamingResponse:
    """Возобновление после подтверждения. confirmed=true → агент работает;
    confirmed=false + correction → новый круг clarify."""
    if not thread_id.strip():
        raise HTTPException(status_code=400, detail="thread_id пуст")
    # Пауза clarify держит буфер живым (он чистится только на done/error).
    # Если буфера нет — turn неизвестен, уже завершён или брошен: резюмить
    # нечего, а Command(resume=...) на «пустом» графе повёл бы себя странно.
    if thread_id not in _agent_trace_buffers:
        raise HTTPException(status_code=409, detail="нет активного turn'а для этого thread_id")
    if len(correction) > _MAX_CORRECTION_LEN:
        raise HTTPException(status_code=400, detail="correction слишком длинный")

    resume = {"confirmed": confirmed, "correction": correction or None}

    async def event_source():
        async for event in _agent_turn_events(
            thread_id, query=None, resume=resume,
        ):
            yield _sse_event(event["type"], event["data"])

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
