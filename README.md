# RAG demo: LM Studio + PostgreSQL/pgvector

Учебный проект для понимания **как работает RAG изнутри**: как создаются
эмбеддинги, как ищется контекст, как собирается финальный промпт для LLM.

Нарочно без LangChain / LlamaIndex — всё в сыром HTTP и SQL, чтобы было
видно, что именно происходит на каждом шаге.

## Что такое RAG за 30 секунд

**RAG** = **R**etrieval **A**ugmented **G**eneration.

LLM не знает деталей вашей базы знаний (внутренней документации, ваших
PDF-ок и так далее). Можно обучить — дорого. Можно при каждом запросе
**подкладывать в промпт релевантный кусок ваших данных** — это и есть RAG.

Pipeline:

```
1) ИНДЕКСАЦИЯ (один раз):
   документы → нарезка на чанки → эмбеддинги → хранение (БД с векторами)

2) ЗАПРОС (на каждый вопрос пользователя):
   вопрос → эмбеддинг вопроса → поиск похожих чанков → промпт с контекстом → LLM → ответ
```

«Эмбеддинг» — это вектор чисел, в котором закодирован смысл текста.
Близкие по смыслу тексты имеют близкие векторы. Сходство меряется обычно
**cosine similarity**.

## Архитектура проекта

```
rag-demo/
├── docker-compose.yml         # Postgres + pgvector
├── .env.example               # настройки (URL LM Studio, токен, DSN БД)
├── requirements.txt           # httpx, psycopg, pgvector, numpy, dotenv
├── docs/                      # корпус документов (.md)
│   ├── python_basics.md
│   ├── http_protocol.md
│   └── databases.md
├── db/
│   └── init.sql               # схема таблицы chunks + HNSW индекс
├── rag/
│   ├── config.py              # настройки из .env
│   ├── chunker.py             # текст → чанки
│   ├── embedder.py            # HTTP к LM Studio /v1/embeddings
│   ├── vector_store.py        # psycopg + pgvector: INSERT / SELECT
│   ├── retriever.py           # склейка: embed(query) → search(top_k)
│   └── generator.py           # HTTP к LM Studio /v1/chat/completions (+stream)
├── web/                       # веб-интерфейс
│   ├── server.py              # FastAPI: /api/ask, /api/ask/stream (SSE)
│   └── static/
│       └── index.html         # одна страница, vanilla JS
├── ingest.py                  # CLI: индексирует docs/
└── ask.py                     # CLI: задаёт вопрос
```

## Как запустить

### TL;DR — всё одной командой

```bash
docker compose up -d --build
```

Поднимет три контейнера:
- `rag_postgres` (pgvector) на :5433
- `rag_backend` (FastAPI + LangGraph-агент) на :8000
- `rag_frontend` (Next.js) на :3000 → http://localhost:3000

Требования:
- `.env` рядом (скопируй `.env.example`).
- LM Studio запущена **на хосте** на :1234 — контейнер ходит туда через
  `host.docker.internal` (на Linux это разрешено через `extra_hosts:
  host-gateway` в compose). Внутрикомпозный `DB_HOST` уже переопределён
  на `postgres` — менять `.env` не нужно.
- Reranker-модель (~570 МБ) скачается при первом обращении в volume
  `hf_cache` — дальше старт мгновенный.

Проиндексировать корпус **внутри** контейнера:
```bash
docker compose exec backend python ingest.py
```

Дальше всё ниже — про **локальный запуск без Docker** (если правишь код
и нужен hot-reload).

### 1. Поднять Postgres

```bash
docker compose up -d postgres
```

Это запустит контейнер `rag_postgres` на порту 5433 и при первом запуске
выполнит `db/init.sql` + `db/init_agent.sql`: создаст расширение `vector`,
таблицы `chunks`/`chats`/`messages`/`agent_messages` и индексы.

Проверить что всё ок:

```bash
docker exec -it rag_postgres psql -U rag -d rag -c "\dx"
docker exec -it rag_postgres psql -U rag -d rag -c "\d chunks"
```

### 2. Подготовить Python-окружение

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# открой .env и проверь URL LM Studio, токен и имя chat-модели
```

Чтобы узнать точные имена моделей, как их видит LM Studio:

```bash
curl -H "Authorization: Bearer $LM_STUDIO_API_KEY" \
     http://192.168.2.129:1234/v1/models | jq
```

### 3. Проиндексировать документы

```bash
python ingest.py
```

Скрипт пройдёт по `docs/`, нарежет каждый файл на чанки, посчитает
эмбеддинги через LM Studio (bge-m3) и положит всё в Postgres.

### 4. Спросить

```bash
python ask.py "Что такое GIL?"
python ask.py --stream "Объясни ACID"
python ask.py --verbose "Как работает HNSW?"
python ask.py -k 3 "Что такое REST?"
```

Самое полезное для понимания — **`--verbose`**: он покажет найденные
чанки с их score'ами и сам промпт, который ушёл в LLM.

### 5. Веб-интерфейс (опционально)

То же самое, но в браузере: видно chunks/prompt/answer на одной странице,
есть переключатель streaming и слайдер top_k.

```bash
uvicorn web.server:app --reload --host 0.0.0.0 --port 8000
```

Открой http://localhost:8000 — там одна страница (без сборщиков, vanilla JS),
с тремя блоками: «что нашли в базе», «итоговый промпт», «ответ LLM».

Что под капотом:
- `web/server.py` — FastAPI с двумя эндпоинтами:
  - `POST /api/ask` — обычный JSON-ответ;
  - `GET /api/ask/stream` — Server-Sent Events, шлёт `meta` (чанки и промпт),
    затем серию `token` событий с кусками ответа, и в конце `done`.
- `web/static/index.html` — один HTML с inline-CSS и vanilla JS.
  Стрим читается стандартным `EventSource` браузера.

OpenAPI-схема и интерактивный Swagger автоматически живут на
http://localhost:8000/docs — удобно тыкать руками.

## Как читать код

Иди по файлам в таком порядке — он совпадает с потоком данных:

1. **`rag/chunker.py`** — самое простое. Скользящее окно по символам с overlap'ом.
2. **`rag/embedder.py`** — HTTP-запрос к `/v1/embeddings`. Видно сырой формат API.
3. **`db/init.sql`** — схема + HNSW индекс под cosine. Прочти комментарии.
4. **`rag/vector_store.py`** — SELECT с оператором `<=>` (cosine distance pgvector).
5. **`rag/retriever.py`** — короткая склейка embedder + store.
6. **`rag/generator.py`** — формирование промпта + streaming SSE.
7. **`ingest.py`** и **`ask.py`** — CLI поверх всего.

## Cosine similarity «руками»

Чтобы было понятно, что именно делает оператор `<=>` в pgvector — вот те
же 3 строки на numpy:

```python
import numpy as np

def cosine_similarity(a, b):
    # Скалярное произведение, делённое на произведение длин векторов.
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# similarity ∈ [-1, 1]
# 1.0 — векторы сонаправлены (тексты «об одном и том же»)
# 0.0 — ортогональны (никак не связаны)
# -1.0 — противоположны (антонимы; на практике почти не встречается)
```

`<=>` в pgvector возвращает `cosine distance = 1 - cosine_similarity` —
поэтому мы пишем `1 - (embedding <=> $query)` чтобы получить именно
similarity.

Почему cosine, а не евклидово расстояние? Эмбеддинги часто
**нормализованы** (длина = 1), и тогда нас интересует только **угол**
между ними, а не их положение в пространстве. Cosine ровно это и меряет.

## Что такое HNSW и зачем он

Самый честный способ найти ближайшие векторы — перебрать **все** строки
и посчитать расстояние до каждой. Это O(N) и при N=10М становится
неприемлемо медленно.

**HNSW** (*Hierarchical Navigable Small World*) — приближённый алгоритм:
строит многослойный граф, где «соседи в графе» = «близкие в пространстве».
Поиск — это жадный спуск по графу, ~O(log N) операций. Платим лёгкой
потерей точности (можем не найти **самый** ближайший вектор, но
с подавляющей вероятностью найдём очень близкий).

В нашей таблице индекс создан так:

```sql
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
```

`vector_cosine_ops` говорит индексу, какую метрику расстояния использовать.
Если хочешь L2 — пиши `vector_l2_ops`, и тогда в SELECT надо использовать
оператор `<->`.

## Полезные команды psql для отладки

Залезть внутрь БД:

```bash
docker exec -it rag_postgres psql -U rag -d rag
```

Посмотреть сколько чанков и из каких файлов:

```sql
SELECT source, COUNT(*) FROM chunks GROUP BY source;
```

Глянуть один эмбеддинг (он длинный — обрежем):

```sql
SELECT source, chunk_index, substring(embedding::text, 1, 80)
FROM chunks LIMIT 1;
```

Поиск руками (нужно подставить любой вектор-запрос — например, скопировать
из БД):

```sql
SELECT source, content, 1 - (embedding <=> '[0.01, 0.02, ...]'::vector) AS sim
FROM chunks
ORDER BY embedding <=> '[0.01, 0.02, ...]'::vector
LIMIT 5;
```

## Что попробовать дальше (если интересно)

- Поменять `CHUNK_SIZE` в `.env` (например, 200 vs 1000) и посмотреть,
  как меняется качество ответов.
- Добавить свой документ в `docs/`, переиндексировать, задать вопрос.
- Поменять `TOP_K` — больше контекста ≠ всегда лучше: бывает «мусор»
  забивает релевантные фрагменты.
- Заменить системный промпт в `rag/generator.py` и посмотреть, как
  меняется стиль ответов.
- Попробовать L2-индекс вместо cosine: пересоздать индекс с
  `vector_l2_ops` и в SQL поменять `<=>` на `<->`. На bge-m3 разница
  маленькая (векторы нормализованы), но методически полезно увидеть.
- Реализовать **reranker**: достать top-20 по cosine, потом
  переупорядочить cross-encoder моделью (например, bge-reranker-v2-m3
  через LM Studio) и взять top-5.
- Полнотекстовый поиск **в дополнение** к векторному (hybrid search):
  Postgres умеет это через `tsvector`. Объединить два рейтинга — даёт
  ощутимый прирост качества.

## Тонкие места, на которые стоит обратить внимание

- **Размерность вектора в схеме (`vector(1024)`) должна совпадать с моделью.**
  Если поменяешь модель (например, на text-embedding-3-small с 1536),
  надо пересоздать таблицу и переиндексировать всё.
- **Эмбеддинги документов и запроса должны считаться одной моделью.**
  Пространства разных моделей несовместимы — иначе поиск выдаст шум.
- **Чанки слишком длинные** → размытый эмбеддинг, плохой поиск.
  **Слишком короткие** → теряется контекст внутри одного чанка.
  Для bge-m3 на русском 400–700 символов обычно работает хорошо.
- **LLM может «галлюцинировать»** даже при хорошем контексте.
  Защита — явные инструкции в system prompt и низкая `temperature`.
