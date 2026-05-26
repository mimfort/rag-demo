"""
vector_store.py — обёртка над PostgreSQL + pgvector.

Что здесь происходит:
- При старте регистрируем тип `vector` в psycopg (`register_vector`),
  чтобы при INSERT можно было передать обычный list[float], а при SELECT
  получать назад тоже list[float] вместо строки.
- Метод upsert_chunks() кладёт чанки в таблицу chunks. Если такой
  (source, chunk_index) уже есть — обновляет (это ON CONFLICT DO UPDATE).
- Метод search() выполняет KNN-поиск: возвращает top_k самых похожих чанков
  по cosine similarity.

Ключевой SQL поиска:

    SELECT source, content, 1 - (embedding <=> %s::vector) AS similarity
    FROM chunks
    ORDER BY embedding <=> %s::vector
    LIMIT %s

Оператор <=> — это cosine distance из pgvector (0 = совпадает, 2 = противоположны).
1 - distance даёт similarity (1 = одинаковые, 0 = ортогональные).
ORDER BY по тому же выражению автоматически использует HNSW-индекс,
если он создан под vector_cosine_ops (а мы его именно так и создали в init.sql).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, replace

import psycopg
from pgvector.psycopg import register_vector

from rag.config import settings


# Разбивает произвольный текст запроса на слова (буквы/цифры в Unicode-смысле).
# Нужен для построения OR-tsquery из пользовательского ввода.
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _build_or_query(query: str) -> str | None:
    """
    «Что такое ACID?» → «Что OR такое OR ACID»

    Зачем: websearch_to_tsquery по умолчанию склеивает слова через AND.
    Для retrieval-фазы это слишком строго — мы хотим вытащить кандидатов,
    которые матчат ХОТЯ БЫ одно слово запроса. Шум потом фильтрует RRF.

    Возвращает None если в запросе нет ни одного слова (пустая строка,
    одни знаки препинания) — text_search в этом случае вернёт пустой список.
    """
    words = _WORD_RE.findall(query)
    if not words:
        return None
    return " OR ".join(words)


# Константа k в формуле Reciprocal Rank Fusion. 60 — каноническое
# значение из оригинальной статьи Cormack/Clarke (2009), хорошо работает
# в большинстве сценариев без подбора.
RRF_K = 60


@dataclass(frozen=True)
class RetrievedChunk:
    """Результат поиска: чанк + его «похожесть» на запрос."""

    source: str
    chunk_index: int
    content: str
    # Cosine similarity с запросом. Заполнено если чанк был найден векторным
    # поиском. В чисто текстовом режиме = 0.0.
    similarity: float
    # Опционально — сам вектор чанка. Нужен для образовательного режима
    # (показать в UI пошаговый расчёт cosine similarity).
    # По умолчанию не запрашивается, чтобы не тащить лишних 4 КБ на каждый чанк.
    embedding: list[float] | None = None

    # --- Поля для hybrid search ---
    # Ранг в векторном поиске (1-based). None — чанк не попал в top-N vector.
    vector_rank: int | None = None
    # Ранг в текстовом поиске (1-based). None — не попал в top-N text.
    text_rank: int | None = None
    # Сырой ts_rank_cd из Postgres. Не сравнимо между запросами по абсолютной
    # величине — только относительно других чанков того же запроса.
    text_score: float | None = None
    # Итоговый RRF-score: чем выше — тем релевантнее в сумме обоих рейтингов.
    rrf_score: float | None = None

    # --- Поля для reranking ---
    # Оценка от cross-encoder / LLM reranker'а, обычно 0..10. None если
    # шаг rerank не делался.
    reranker_score: float | None = None
    # Позиция чанка ДО reranking'а (1-based в списке, который пришёл от
    # hybrid/vector/text). Нужна UI чтобы показать «#3 → #1».
    original_rank: int | None = None
    # --- Поля для MMR ---
    # Позиция в MMR-отборе (1-based). None если MMR не применялся.
    # Нужна чтобы показать в UI как MMR переставил порядок.
    mmr_rank: int | None = None


@dataclass(frozen=True)
class ScoredChunk:
    """Лёгкая версия для гистограммы similarity по всей базе (без content)."""

    source: str
    chunk_index: int
    similarity: float


@dataclass(frozen=True)
class TextHit:
    """Результат текстового поиска (BM25-подобный rank)."""

    source: str
    chunk_index: int
    content: str
    text_score: float


@dataclass(frozen=True)
class ChunkInput:
    """Чанк, готовый к загрузке в БД."""

    source: str
    chunk_index: int
    content: str
    embedding: list[float]


class VectorStore:
    """
    Тонкая обёртка над таблицей chunks. Открывает одно соединение к Postgres
    и переиспользует его на протяжении жизни объекта.
    """

    def __init__(self, dsn: str | None = None) -> None:
        # autocommit=True избавляет от ручного commit/rollback.
        # Для нашего сценария (батчевый INSERT и поиск) это удобно;
        # в продакшене обычно делают наоборот — управляют транзакциями явно.
        self._conn = psycopg.connect(dsn or settings.db_dsn, autocommit=True)
        # ВАЖНО: регистрируем pgvector в текущем соединении.
        # Без этой строки psycopg будет считать колонку embedding обычной строкой
        # и нам пришлось бы вручную форматировать векторы как "[0.1, 0.2, ...]".
        register_vector(self._conn)
        # Идемпотентная миграция: добавляем то, чего ещё нет.
        # Удобно для апгрейда уже работающей БД без пересоздания тома.
        self._ensure_schema()

    # SQL-выражение GENERATED-колонки tsv. Хранится в одном месте,
    # чтобы и при создании, и при проверке «устарела ли колонка»
    # сравнивать с одной и той же строкой.
    _TSV_EXPR = (
        "to_tsvector('russian'::regconfig, content) || "
        "to_tsvector('simple'::regconfig, content)"
    )

    def _ensure_schema(self) -> None:
        """
        Self-healing миграция: добавляет недостающие объекты и подгоняет
        существующие под текущую схему.

        Особенность для нашего проекта: колонка `tsv` была сначала создана
        как `to_tsvector('russian', content)`, но русский стеммер плохо
        работает с английскими аббревиатурами (ACID, GIL). Поэтому добавили
        ещё `simple` через `||`. ALTER TABLE не умеет менять выражение
        GENERATED-колонки — приходится DROP + ADD. Postgres сам пересчитает
        значения для всех существующих строк.

        В реальном проекте такие изменения делают отдельным инструментом
        (alembic, dbmate, sqitch) — там есть журнал применённых, rollback.
        Для учебного демо такой минимализм оправдан.
        """
        with self._conn.cursor() as cur:
            # 1. Узнаём текущее выражение колонки tsv (если она вообще есть).
            cur.execute(
                """
                SELECT pg_get_expr(d.adbin, d.adrelid)
                FROM pg_attribute a
                LEFT JOIN pg_attrdef d
                  ON d.adrelid = a.attrelid AND d.adnum = a.attnum
                WHERE a.attrelid = 'chunks'::regclass
                  AND a.attname  = 'tsv'
                  AND a.attnum > 0
                """
            )
            row = cur.fetchone()
            current_expr: str | None = row[0] if row else None

            # 2. Решаем: создать, пересоздать или ничего не делать.
            needs_recreate = (
                current_expr is not None
                and "'simple'" not in current_expr
            )

            if needs_recreate:
                # Старая версия (только 'russian') — пересоздаём.
                # DROP COLUMN автоматически снимает зависящий индекс.
                cur.execute("ALTER TABLE chunks DROP COLUMN tsv")
                current_expr = None  # дальше пойдём в ветку «добавить»

            if current_expr is None:
                cur.execute(
                    f"ALTER TABLE chunks "
                    f"ADD COLUMN tsv tsvector "
                    f"GENERATED ALWAYS AS ({self._TSV_EXPR}) STORED"
                )

            # 3. GIN-индекс — нужен для быстрого `tsv @@ tsquery`.
            #    IF NOT EXISTS делает вызов идемпотентным.
            cur.execute(
                "CREATE INDEX IF NOT EXISTS chunks_tsv_idx "
                "ON chunks USING GIN (tsv)"
            )

    # --- запись ---

    def delete_source(self, source: str) -> int:
        """
        Удаляет все чанки конкретного файла.
        Используется при переиндексации: проще удалить и вставить заново,
        чем пытаться определить, какие чанки изменились.
        Возвращает количество удалённых строк.
        """
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source = %s", (source,))
            return cur.rowcount

    def insert_chunks(self, chunks: list[ChunkInput]) -> None:
        """
        Пакетная вставка чанков.
        executemany эффективнее цикла из execute: psycopg отправит
        все строки одним запросом по сети.

        ВНИМАНИЕ к плейсхолдеру %s. В psycopg это позиционный плейсхолдер
        (не %s из printf!) — драйвер сам экранирует и подставит значения,
        SQL-инъекции невозможны.
        """
        if not chunks:
            return

        sql = """
            INSERT INTO chunks (source, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (source, chunk_index) DO UPDATE
            SET content = EXCLUDED.content,
                embedding = EXCLUDED.embedding
        """
        rows = [
            (c.source, c.chunk_index, c.content, c.embedding)
            for c in chunks
        ]
        with self._conn.cursor() as cur:
            cur.executemany(sql, rows)

    def count(self) -> int:
        """Сколько всего чанков в базе."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    # --- чтение ---

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        include_embeddings: bool = False,
    ) -> list[RetrievedChunk]:
        """
        Возвращает top_k наиболее похожих чанков на query_embedding.

        Как читать запрос:
        - `embedding <=> %s::vector` — cosine distance между нашей колонкой
          и переданным вектором запроса.
        - `1 - distance` — similarity (более интуитивно: чем больше, тем лучше).
        - ORDER BY по distance сортирует от самого близкого к самому далёкому.
        - LIMIT ограничивает количество возвращаемых строк.

        Postgres с HNSW-индексом выполнит это эффективно даже на больших
        таблицах: он не будет считать distance для каждой строки,
        а пройдёт по графу ближайших соседей.

        include_embeddings=True добавит сам вектор каждого чанка в результат.
        Нужно для UI-разъяснения «как считается cosine руками».
        """
        # Колонку embedding выбираем только если попросили — экономим трафик.
        embed_select = ", embedding" if include_embeddings else ""
        sql = f"""
            SELECT
                source,
                chunk_index,
                content,
                1 - (embedding <=> %s::vector) AS similarity
                {embed_select}
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (query_embedding, query_embedding, top_k))
            rows = cur.fetchall()

        result: list[RetrievedChunk] = []
        for rank, row in enumerate(rows, start=1):
            embedding = None
            if include_embeddings:
                # pgvector через register_vector возвращает numpy.ndarray —
                # приводим к обычному list[float], чтобы потом легко
                # сериализовать в JSON.
                embedding = list(row[4])
            result.append(
                RetrievedChunk(
                    source=row[0],
                    chunk_index=int(row[1]),
                    content=row[2],
                    similarity=float(row[3]),
                    embedding=embedding,
                    vector_rank=rank,
                )
            )
        return result

    # ----------------------------------------------------------------
    # Полнотекстовый поиск
    # ----------------------------------------------------------------
    def text_search(self, query: str, top_n: int = 20) -> list[TextHit]:
        """
        Полнотекстовый поиск по колонке tsv.

        Логика построения tsquery:
        1. Разбиваем пользовательский ввод на слова и склеиваем через OR
           (см. _build_or_query). Без этого websearch_to_tsquery поставил
           бы AND по умолчанию и пустую выдачу на «Что такое ACID?».
        2. Парсим OR-строку двумя конфигурациями:
             - 'russian' — стеммит русские слова;
             - 'simple'  — оставляет токены как есть, нужно для англ.
                          аббревиатур (ACID, GIL).
           И объединяем tsquery оператором `||` (OR на tsquery'ях).
        3. Это симметрично к колонке tsv (см. _ensure_schema): и индекс,
           и запрос покрывают обе конфигурации.

        Прочие операторы:
        - `tsv @@ q` — «текст матчит запрос?». Использует GIN-индекс.
        - `ts_rank_cd` — cover density ranking, учитывает близость слов.
          Лучше базового ts_rank для коротких фраз.

        Если в query нет ни одного слова — возвращаем пустой список
        не дёргая БД.
        """
        or_query = _build_or_query(query)
        if or_query is None:
            return []

        sql = """
            WITH q AS (
                SELECT
                    websearch_to_tsquery('russian', %s) ||
                    websearch_to_tsquery('simple',  %s) AS query
            )
            SELECT
                source,
                chunk_index,
                content,
                ts_rank_cd(tsv, q.query) AS rank
            FROM chunks, q
            WHERE tsv @@ q.query
            ORDER BY rank DESC
            LIMIT %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (or_query, or_query, top_n))
            rows = cur.fetchall()
        return [
            TextHit(
                source=row[0],
                chunk_index=int(row[1]),
                content=row[2],
                text_score=float(row[3]),
            )
            for row in rows
        ]

    # ----------------------------------------------------------------
    # Hybrid search (RRF)
    # ----------------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        candidate_n: int = 20,
    ) -> list[RetrievedChunk]:
        """
        Hybrid search через Reciprocal Rank Fusion.

        Шаги:
          1. Vector search: top candidate_n по cosine.
          2. Text search:   top candidate_n по ts_rank_cd.
          3. Для каждого встреченного чанка считаем:
                rrf_score = Σ 1 / (RRF_K + rank_i)
             по тем рейтингам, где он встретился.
          4. Сортируем по rrf_score, берём top_k.

        Почему RRF, а не «weighted sum» нормализованных score:
          - score'ы из разных систем плохо нормализуются (cosine ∈ [0,1],
            ts_rank — какое-то положительное число неизвестного потолка).
          - RRF использует только РАНГИ — самый устойчивый сигнал.
          - Не требует подбора весов. По умолчанию k=60 годится для всего.

        candidate_n больше top_k намеренно: чем шире «воронка», тем больше
        шанс что слабый-в-обоих-но-стабильный кандидат поднимется в финал.
        """
        # 1. Два поиска параллельно делать не будем — на 21 чанке смысла нет.
        vec_hits = self.search(query_embedding, top_k=candidate_n)
        txt_hits = self.text_search(query, top_n=candidate_n)

        # 2. Складываем всё в словарь по ключу (source, chunk_index).
        #    В нём же копим RRF-score и метаданные обоих рейтингов.
        by_key: dict[tuple[str, int], RetrievedChunk] = {}
        rrf_scores: dict[tuple[str, int], float] = defaultdict(float)

        for hit in vec_hits:
            key = (hit.source, hit.chunk_index)
            by_key[key] = hit
            # hit.vector_rank уже выставлен в search(); используем его.
            rrf_scores[key] += 1.0 / (RRF_K + hit.vector_rank)

        for rank, hit in enumerate(txt_hits, start=1):
            key = (hit.source, hit.chunk_index)
            if key in by_key:
                # Кандидат был и в векторном — обогащаем текстовыми полями.
                existing = by_key[key]
                by_key[key] = replace(
                    existing,
                    text_rank=rank,
                    text_score=hit.text_score,
                )
            else:
                # Только текстовый — vector_rank остаётся None, similarity=0.
                by_key[key] = RetrievedChunk(
                    source=hit.source,
                    chunk_index=hit.chunk_index,
                    content=hit.content,
                    similarity=0.0,
                    text_rank=rank,
                    text_score=hit.text_score,
                )
            rrf_scores[key] += 1.0 / (RRF_K + rank)

        # 3. Проставляем rrf_score в каждый чанк и сортируем.
        result: list[RetrievedChunk] = []
        for key, chunk in by_key.items():
            result.append(replace(chunk, rrf_score=rrf_scores[key]))
        result.sort(key=lambda c: c.rrf_score or 0.0, reverse=True)

        return result[:top_k]

    def get_embeddings(
        self, keys: list[tuple[str, int]]
    ) -> dict[tuple[str, int], list[float]]:
        """
        Батчевая загрузка эмбеддингов нескольких чанков.

        Принимает список (source, chunk_index) и возвращает словарь
        ключ → вектор. Используется для MMR — там нужны эмбеддинги
        всех кандидатов одним запросом, чтобы считать попарную
        cosine similarity.

        Реализовано через ANY(...) с массивами — psycopg сам конвертирует
        list of tuples. Один SQL вместо N запросов.
        """
        if not keys:
            return {}
        # Postgres не принимает напрямую массив кортежей в ANY,
        # поэтому раскладываем на два параллельных массива.
        sources = [k[0] for k in keys]
        indices = [k[1] for k in keys]
        sql = """
            SELECT source, chunk_index, embedding
            FROM chunks
            WHERE (source, chunk_index) IN (
                SELECT unnest(%s::text[]), unnest(%s::int[])
            )
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (sources, indices))
            rows = cur.fetchall()
        return {
            (row[0], int(row[1])): list(row[2])
            for row in rows
        }

    def get_neighbors(
        self,
        centers: list[tuple[str, int]],
        radius: int = 1,
    ) -> dict[tuple[str, int], str]:
        """
        Для каждого «центрального» чанка возвращает соседей в радиусе
        `radius` чанков (по chunk_index). Используется для расширения
        контекста при формировании промпта LLM: смысл часто разрывается
        на границах чанков, а соседи восстанавливают связность.

        Возвращает dict: (source, chunk_index) → content. Включает сами
        центральные чанки тоже — это удобно при сборке итогового текста
        (не надо отдельно их подтягивать).

        Реализация — один SQL-запрос на всё. Для каждого центра считаем
        диапазон [center-radius, center+radius] и берём все чанки этого
        source с chunk_index в диапазоне.
        """
        if not centers:
            return {}

        # Группируем по source и складываем все нужные индексы.
        # Дубликаты (если соседи разных центров совпадают) уберём естественно
        # через UNION в SQL.
        wanted: set[tuple[str, int]] = set()
        for src, idx in centers:
            for delta in range(-radius, radius + 1):
                wanted.add((src, idx + delta))

        sources = [k[0] for k in wanted]
        indices = [k[1] for k in wanted]
        sql = """
            SELECT source, chunk_index, content
            FROM chunks
            WHERE (source, chunk_index) IN (
                SELECT unnest(%s::text[]), unnest(%s::int[])
            )
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (sources, indices))
            rows = cur.fetchall()
        return {(row[0], int(row[1])): row[2] for row in rows}

    def get_embedding(self, source: str, chunk_index: int) -> list[float] | None:
        """
        Возвращает embedding-вектор одного чанка. Нужен когда мы попали
        в чанк нечасто (через text/hybrid) и потом хотим показать его в
        образовательном расчёте cosine.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT embedding FROM chunks "
                "WHERE source = %s AND chunk_index = %s",
                (source, chunk_index),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return list(row[0])

    def score_all(self, query_embedding: list[float]) -> list[ScoredChunk]:
        """
        Возвращает similarity для ВСЕХ чанков в базе — без LIMIT.
        Полезно для визуализации «как ранжируется вся база на этот запрос»:
        видно разрыв между релевантными (~0.7+) и нерелевантными (~0.3) чанками.

        На большой базе так делать нельзя (это полный скан), но для учебных
        21 чанка — копейки.
        """
        sql = """
            SELECT
                source,
                chunk_index,
                1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            ORDER BY similarity DESC
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (query_embedding,))
            rows = cur.fetchall()
        return [
            ScoredChunk(
                source=row[0],
                chunk_index=int(row[1]),
                similarity=float(row[2]),
            )
            for row in rows
        ]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
