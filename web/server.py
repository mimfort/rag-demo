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

import json
import math
import time
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.config import settings
from rag.embedder import LMStudioEmbedder
from rag.generator import LMStudioGenerator, build_user_prompt
from rag.vector_store import RetrievedChunk, ScoredChunk, VectorStore


# Допустимые режимы поиска. Используется и в API-моделях, и в логике.
SEARCH_MODES = ("vector", "text", "hybrid")


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

# Эти объекты заполнятся в startup-хендлере ниже.
_embedder: LMStudioEmbedder | None = None
_store: VectorStore | None = None
_generator: LMStudioGenerator | None = None


@app.on_event("startup")
def _startup() -> None:
    """Открываем долгоживущие соединения при старте процесса."""
    global _embedder, _store, _generator
    _embedder = LMStudioEmbedder()
    _store = VectorStore()
    _generator = LMStudioGenerator()


@app.on_event("shutdown")
def _shutdown() -> None:
    """Аккуратно закрываем всё при остановке сервера."""
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
        )
        for c in chunks
    ]


# Сколько первых значений вектора показывать в UI. Полные 1024 не нужны:
# для визуализации достаточно ~64 — глаз и так не различит больше.
PREVIEW_DIMS = 64


def _retrieve_with_explain(
    query: str,
    top_k: int,
    min_similarity: float = 0.0,
    search_mode: str = "hybrid",
) -> tuple[list[RetrievedChunk], ExplainOut]:
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

    # 1. Эмбеддинг запроса считаем всегда — для образовательных панелей.
    t0 = time.perf_counter()
    query_vec = _embedder.embed_one(query)
    embed_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Поиск в выбранном режиме.
    #    Для top-1 нам нужен embedding чанка чтобы посчитать cosine руками —
    #    но _store.hybrid_search / text_search его не возвращают.
    #    Поэтому отдельным запросом достаём вектор top-1 чанка ниже.
    if search_mode == "vector":
        chunks = _store.search(query_vec, top_k=top_k, include_embeddings=True)
    elif search_mode == "text":
        text_hits = _store.text_search(query, top_n=top_k)
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
            query, query_vec, top_k=top_k, candidate_n=20
        )

    # 3. Similarity для ВСЕХ чанков базы — для гистограммы в UI.
    #    Не зависит от search_mode: гистограмма всегда показывает cosine,
    #    это «карта семантической близости» базы к запросу.
    all_scores = _store.score_all(query_vec)

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
        embed_model=settings.embedding_model,
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
    )
    return chunks, explain


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "chunks_in_db": _store.count() if _store else 0}


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Обычный (нестриминговый) RAG-запрос."""
    assert _generator is not None  # для тайп-чекера

    chunks, explain = _retrieve_with_explain(
        req.query, req.top_k, req.min_similarity, req.search_mode,
    )
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="В базе нет чанков. Запусти `python ingest.py`.",
        )
    prompt = build_user_prompt(req.query, chunks)
    answer = _generator.generate(req.query, chunks)
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

    chunks, explain = _retrieve_with_explain(
        query, top_k, min_similarity, search_mode,
    )

    def event_source() -> Iterator[str]:
        # 1. Сначала отдаём «meta»: что нашли, промпт и explain-блок.
        #    Используем _to_chunk_out (тот же, что в POST /api/ask) — иначе
        #    легко забыть какое-нибудь поле когда схема расширяется.
        meta_payload = {
            "chunks": [c.model_dump() for c in _to_chunk_out(chunks)],
            "prompt": build_user_prompt(query, chunks),
            # Pydantic.model_dump() даёт обычный dict — его легко уложить в JSON.
            "explain": explain.model_dump(),
        }
        yield _sse_event("meta", meta_payload)

        # Если контекста нет — заканчиваем, не дёргая LLM.
        if not chunks:
            yield _sse_event(
                "token",
                {"text": "В базе нет чанков. Запусти `python ingest.py`."},
            )
            yield _sse_event("done", {})
            return

        # 2. Стримим токены ответа.
        for piece in _generator.generate_stream(query, chunks):
            yield _sse_event("token", {"text": piece})

        # 3. Сигнал конца.
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
