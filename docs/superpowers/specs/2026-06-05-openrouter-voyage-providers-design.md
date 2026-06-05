# Миграция провайдеров: OpenRouter (LLM) + Voyage AI (embeddings, rerank)

Дата: 2026-06-05

## Цель

Перевести проект с локальной LM Studio на сетевые провайдеры:

- **Chat-LLM** → OpenRouter (OpenAI-совместимый API).
- **Embeddings** → Voyage AI (`voyage-4-large`, 1024 dim).
- **Reranker** → Voyage AI (`rerank-2.5`), вместо локального cross-encoder.

Ключевые требования: провайдер-агностичность (лёгкая смена оператора),
безопасный и качественный код, сохранение стриминга.

## Решения (зафиксированы при брейншторме)

| Вопрос | Решение |
|---|---|
| Embedding-модель | `voyage-4-large`, 1024 dim (схема БД `vector(1024)` без изменений) |
| Rerank-модель | `rerank-2.5` (сетевой, заменяет локальный cross-encoder) |
| Локальный cross-encoder | Удалить (torch / sentence-transformers выкинуть из requirements) |
| Глубина абстракции | Прагматичная: OpenAI-совместимый транспорт для chat + Protocol-интерфейсы для embed/rerank |
| Имена env для chat | Нейтральные `LLM_*` (а не `OPENROUTER_*`) — лучше для смены оператора |
| Обратная совместимость со старыми env | Нет — чистая замена `LM_STUDIO_*` |
| `LMStudioGenerator` | Переименовать в `ChatGenerator` по всем импортам |

## Архитектура

### Текущее состояние

- Chat идёт двумя путями: `agent/llm.py::make_llm()` (`ChatOpenAI` для LangGraph-агента)
  и `rag/generator.py::LMStudioGenerator` (httpx к `/chat/completions`, используется
  RAG + `rewriter`/`router`/`decomposer`/`history`). Оба бьют в LM Studio.
- Embeddings: `rag/embedder.py::LMStudioEmbedder` → `/embeddings` LM Studio.
- Reranker: `rag/reranker.py::CrossEncoderReranker` → локальный sentence-transformers.

### Целевое состояние

Три роли провайдеров, каждая со своим seam'ом для смены оператора:

1. **Chat (LLM)** — OpenAI-совместимый транспорт переиспользуется как есть; OpenRouter
   и LM Studio отличаются лишь значениями в `.env`. Смена оператора = правка `.env`.
2. **Embeddings** — `Protocol Embedder` + реализация `VoyageEmbedder` + фабрика `make_embedder()`.
3. **Rerank** — `Protocol Reranker` + реализация `VoyageReranker` + фабрика `make_reranker()`.

## Конфигурация (.env и rag/config.py)

Новый набор переменных (`.env`, `.env.example`):

```dotenv
# --- LLM (chat) — OpenAI-совместимый провайдер (OpenRouter по умолчанию) ---
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
LLM_MODEL=                       # пользователь указывает модель, напр. anthropic/claude-... или openai/gpt-...
LLM_HTTP_REFERER=                # опц. заголовок атрибуции OpenRouter
LLM_APP_TITLE=                   # опц. заголовок атрибуции OpenRouter

# --- Voyage AI (embeddings + rerank) ---
VOYAGE_API_KEY=pa-...
VOYAGE_BASE_URL=https://api.voyageai.com/v1
VOYAGE_EMBEDDING_MODEL=voyage-4-large
VOYAGE_RERANK_MODEL=rerank-2.5
EMBEDDING_DIM=1024
```

`Settings` (frozen dataclass) получает поля:
`llm_base_url`, `llm_api_key`, `llm_model`, `llm_http_referer` (опц.), `llm_app_title` (опц.),
`voyage_api_key`, `voyage_base_url`, `voyage_embedding_model`, `voyage_rerank_model`.
`embedding_dim` остаётся. Поля `lm_studio_*` и `chat_model` удаляются.

Безопасность: ключи только из окружения, в DSN/логи не попадают. `_env()` падает при
отсутствии обязательной переменной с понятным сообщением (как сейчас).

## Компоненты

### Chat: agent/llm.py + rag/generator.py

- `make_llm()` — `ChatOpenAI(model=llm_model, base_url=llm_base_url, api_key=llm_api_key,
  temperature=0.2, max_retries=1)`. Стриминг агента через `astream_events` работает нативно.
- `ChatGenerator` (переименование `LMStudioGenerator`) — httpx-клиент к `/chat/completions`:
  - `generate()` (нестрим) и `generate_stream()` (SSE) — логика без изменений; OpenRouter
    шлёт тот же формат `data: {...}\n\n` + `data: [DONE]`.
  - Опциональные заголовки `HTTP-Referer` и `X-Title` добавляются, если заданы в конфиге.
  - Конструктор читает `llm_*` из настроек.
- Обновить импорты `LMStudioGenerator → ChatGenerator` в: `ask.py`, `web/server.py`,
  `rag/rewriter.py`, `rag/router.py`, `rag/decomposer.py`, `rag/history.py`.

### Embeddings: rag/embedder.py

- `Protocol Embedder`:
  - `embed_query(text: str) -> list[float]` — `input_type="query"`.
  - `embed_documents(texts: Iterable[str]) -> list[list[float]]` — `input_type="document"`.
- `VoyageEmbedder` — httpx к `POST {base}/embeddings`, тело `{model, input, input_type}`:
  - Батч ≤ 1000 текстов за запрос.
  - Проверка размерности каждого вектора == `embedding_dim`, иначе понятная ошибка.
  - Сортировка `data` по `index` (как и раньше).
  - keep-alive httpx-клиент, `close()` / контекст-менеджер.
- `make_embedder() -> Embedder` — единственная точка выбора провайдера.

Замена `input_type` — осознанное улучшение: Voyage даёт прирост ретрива от разделения
query/document, чего старый `embed_one/embed_many` не делал.

Call-sites:
- `ingest.py` — `embed_documents(...)` при индексации.
- `rag/retriever.py` — `embed_query(...)` для запроса.
- `evals/runner.py` — соответственно query/documents через `make_embedder()`.

### Reranker: rag/reranker.py

- `Protocol Reranker`: `rerank(query, chunks, top_k=None) -> list[RetrievedChunk]`
  (сигнатура сохранена; обогащение `reranker_score` / `original_rank` остаётся —
  UI и evals не ломаются).
- `VoyageReranker` — httpx к `POST {base}/rerank`, тело `{model, query, documents, top_k}`:
  - `documents` = `[c.content for c in chunks]` (≤ 1000).
  - Ответ: список `{index, relevance_score}` — маппим `index` → исходный чанк,
    проставляем `reranker_score` и `original_rank`, сортируем по убыванию score.
  - При пустом `chunks` — возврат `[]` без сетевого вызова.
- `make_reranker() -> Reranker`.
- Удаляем `CrossEncoderReranker`. `web/server.py` и `evals/runner.py` → `make_reranker()`.

## Зависимости и схема БД

- `requirements.txt`: удалить `sentence-transformers` (и транзитивный torch).
  `httpx` и `langchain-openai` остаются. Voyage гоняем тем же httpx — без отдельного SDK,
  в духе проекта (видно сырое HTTP-взаимодействие).
- Схема БД `vector(1024)` не меняется.
- **Переиндексация обязательна**: после смены embed-модели старые bge-m3-векторы
  несовместимы с voyage — нужно прогнать `ingest.py` заново. Отметить в README.

## Обработка ошибок

- `raise_for_status()` на всех httpx-вызовах — падаем громко при 4xx/5xx, чтобы не
  работать с битыми данными.
- Понятные сообщения при несовпадении размерности эмбеддинга и при отсутствии env.
- Таймауты: embeddings ~60s, chat ~120s (стрим), rerank ~60s — как сейчас.

## Тестирование

- `VoyageEmbedder`: мок-httpx — корректный разбор `data`, сортировка по `index`,
  ошибка при неверной размерности, правильный `input_type` в payload.
- `VoyageReranker`: мок-httpx — маппинг `index`→чанк, проставление score/rank,
  сортировка по убыванию, пустой вход без вызова сети.
- `ChatGenerator`: payload нестрим/стрим, парсинг SSE, наличие заголовков атрибуции
  при заданном конфиге и их отсутствие — при пустом.

## Вне scope

- Полный registry-фреймворк провайдеров (выбран прагматичный вариант).
- Поддержка `output_dimension` ≠ 1024 (используем дефолт voyage-4-large).
- Кэширование эмбеддингов/реранка.
