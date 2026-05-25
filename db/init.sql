-- =============================================================================
-- init.sql — выполняется ОДИН РАЗ при первом старте контейнера postgres
-- (docker-entrypoint-initdb.d запускает все *.sql из этой папки).
-- =============================================================================

-- Включаем расширение pgvector. После этого Postgres понимает тип `vector`
-- и операторы расстояния:
--   <->  L2 (евклидово)
--   <#>  отрицательное скалярное произведение
--   <=>  cosine distance  (нам нужно именно это)
CREATE EXTENSION IF NOT EXISTS vector;

-- Таблица для хранения нарезанных кусков (чанков) документов.
-- Это и есть наш «векторный индекс»: одна строка = один чанк + его эмбеддинг.
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,

    -- Откуда чанк (имя исходного файла). Полезно показывать в ответе
    -- LLM как «источник», и удобно удалять/переиндексировать по файлу.
    source      TEXT NOT NULL,

    -- Порядковый номер чанка внутри документа.
    -- В паре (source, chunk_index) даёт уникальный идентификатор куска.
    chunk_index INT  NOT NULL,

    -- Сам текст чанка — то, что мы будем подкладывать в промпт LLM.
    content     TEXT NOT NULL,

    -- Эмбеддинг — числовой вектор смысла этого чанка.
    -- 1024 — это размерность модели bge-m3.
    -- ВНИМАНИЕ: число должно совпадать с EMBEDDING_DIM в .env и
    -- размерностью векторов, которые возвращает модель.
    embedding   vector(1024) NOT NULL,

    -- Полнотекстовый индекс для BM25-подобного поиска (hybrid search).
    -- GENERATED ALWAYS AS ... STORED — это «вычисляемая колонка»: Postgres
    -- сам пересчитывает её при INSERT/UPDATE content. Не надо триггеров,
    -- не надо думать про синхронизацию.
    --
    -- Комбинируем ДВЕ конфигурации через оператор `||` (конкатенация tsvector):
    --   'russian' — стемминг русских слов («индексы» → «индекс»);
    --   'simple'  — без стемминга, lowercase. Ловит англ-аббревиатуры
    --               (ACID, GIL, HTTP) точным совпадением, для которых
    --               русский стеммер бесполезен.
    -- В запросе тоже используем обе — см. text_search() в rag/vector_store.py.
    tsv         tsvector
                  GENERATED ALWAYS AS (
                    to_tsvector('russian', content) ||
                    to_tsvector('simple', content)
                  ) STORED,

    UNIQUE (source, chunk_index)
);

-- HNSW-индекс по cosine distance.
--
-- Без индекса pgvector делает полный перебор всех строк — медленно при
-- сотнях тысяч чанков, но абсолютно точно. С HNSW поиск идёт по графу
-- ближайших соседей: чуть менее точно (approximate nearest neighbors),
-- зато логарифмически быстро.
--
-- `vector_cosine_ops` говорит индексу: «строй меня под cosine distance»
-- (есть ещё vector_l2_ops и vector_ip_ops под другие метрики).
--
-- Альтернатива HNSW — ivfflat. HNSW обычно качественнее по умолчанию,
-- но требует чуть больше памяти. Для нашего демо разница не видна.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks
    USING hnsw (embedding vector_cosine_ops);

-- Индекс по source — чтобы быстро удалять все чанки одного документа
-- при переиндексации (DELETE FROM chunks WHERE source = ...).
CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source);

-- GIN-индекс по tsv. GIN — Generalized Inverted Index, идеально подходит
-- для полнотекстового поиска: для каждого слова хранит список документов,
-- в которых оно встречается. На запрос `tsv @@ tsquery` Postgres быстро
-- вытаскивает кандидатов из инвертированного индекса.
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
