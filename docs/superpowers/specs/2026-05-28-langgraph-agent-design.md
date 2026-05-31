# LangGraph-агент: «Спорт-консьерж Рондо»

Дата: 2026-05-28
Статус: предложение, ожидает финального ревью

## Зачем

После того как TG-инициатива была свёрнута, нужен другой осмысленный
кейс для практики LangGraph поверх существующего RAG-стека. Выбран
**ReAct-агент для СКК «Рондо»** — публичный API
`https://api.skkrondo.ru` без auth, с реальными use-case'ами:
проверка свободных кортов и погоды на дату/диапазон.

Главный пользовательский запрос-кейс: **«найди солнечный день на
следующей неделе со свободным кортом»** — связка нескольких tool-вызовов
и синтез ответа из накопленных данных.

Проект остаётся образовательным: цель — пройти ключевые LangGraph-
паттерны (StateGraph, ToolNode, conditional edges, loops) на маленькой
локальной модели (Gemma 4 e2b в LM Studio).

## Контекст и ограничения

- **Локальный single-user setup**, как и весь остальной проект.
- **LLM = Gemma 4 e2b** (2B параметров) через LM Studio (OpenAI-compat).
  Маленькая модель — native function-calling может быть нестабильным.
  Это **сознательный выбор** ради учебы; mitigation'ы — в разделе
  «Известные риски».
- **Read-only**: используем только GET-эндпоинты skkrondo. Booking
  (POST/DELETE) НЕ трогаем — без auth это опасно.
- **Один tool-набор в Spec 1**: `get_weather` и `get_courts_availability`.
  Другие endpoints (table tennis, sports club, events, afisha, RAG-FAQ
  поверх регламента) — отдельные spec'и.

## Архитектура

Новый Python-модуль `agent/` рядом с `rag/`:

| Файл              | Ответственность                                                              |
|-------------------|------------------------------------------------------------------------------|
| `agent/config.py` | `SKKRONDO_BASE_URL`, `MAX_ITER`, тайм-ауты HTTP-клиента                      |
| `agent/state.py`  | `AgentState(TypedDict)` — единственное поле `messages: list[BaseMessage]`    |
| `agent/tools.py`  | Tools `get_weather`, `get_courts_availability` (async httpx + парсинг)       |
| `agent/llm.py`    | Фабрика `ChatOpenAI` поверх LM Studio, `bind_tools(...)`                     |
| `agent/graph.py`  | `StateGraph` с узлами `agent` + `tools` и `tools_condition` edge'ом          |
| `agent/runner.py` | Entry-point с `graph.astream(stream_mode="updates")`, конвертация в события  |

### LangGraph-граф (классический ReAct)

```
       ┌────────┐
       │  agent │ ◄────┐
       └───┬────┘      │
           │           │
    tool_calls?        │
     ┌─────┴─────┐     │
     │           │     │
     ▼           ▼     │
  no tools   tool_calls│
     │           │     │
     ▼           ▼     │
   [END]    ┌────────┐ │
            │ tools  │─┘
            └────────┘
```

- `agent` узел: вызывает LLM с накопленными `messages` + bound tools.
  Возвращает либо `AIMessage` без tool_calls (→ END), либо c tool_calls
  (→ `tools` node).
- `tools` узел: `ToolNode(TOOLS)` из `langgraph.prebuilt` исполняет
  tool_calls и добавляет `ToolMessage`'ы в state.
- Conditional edge между `agent` и `tools/END` через `tools_condition`.
- Loop: `agent → tools → agent → tools → … → END`, с лимитом `MAX_ITER=10`
  как safety net.

### Tools

```python
@tool
async def get_weather(date: str) -> dict:
    """Прогноз погоды на дату YYYY-MM-DD."""
    # GET https://api.skkrondo.ru/weather/weather/{date}
    # → {date, temperature_c, condition, sunny: bool, raw}

@tool
async def get_courts_availability(date: str) -> dict:
    """Занятые слоты кортов на дату YYYY-MM-DD."""
    # GET https://api.skkrondo.ru/court_reservations/all/{date}
    # → {date, reservations: [...], summary: "текст"}
```

Каждая возвращает не сырой JSON, а **компактный человекочитаемый
dict** — это снижает нагрузку на контекст LLM. `_classify_sunny` —
простая текстовая эвристика по полю `condition` (ясно / малооблачно →
True; пасмурно / дождь → False). `_summarize_courts` сжимает массив
броней в одну строку.

### LLM-фабрика

```python
ChatOpenAI(
    model=rag_settings.chat_model,            # gemma-4-e2b
    base_url=rag_settings.lm_studio_base_url,
    api_key=rag_settings.lm_studio_api_key,
    temperature=0.2,
    max_retries=1,
)
```

LM Studio полностью OpenAI-compatible — `ChatOpenAI` работает без
модификаций. `bind_tools(TOOLS)` шлёт descriptions в формате, который
LM Studio пробрасывает в Gemma.

### Системный промпт

```
Ты — помощник по СКК «Рондо». Ты можешь вызывать tools чтобы получить
погоду и список забронированных кортов на конкретные даты.

Сегодня: {today}.

Если пользователь спрашивает про «свободный день», «следующую неделю»,
«погоду» — определи диапазон дат и проверь их tool'ами. Когда соберёшь
достаточно данных — отвечай по-русски, упоминай конкретные даты и часы.
```

Инжектируется один раз в начало `messages` при первом обращении к
`agent`-узлу. `{today}` подставляется текущей датой через `date.today().isoformat()`.

## API

### Endpoints

| Method | Path                          | Body / Query     | Returns                                          |
|--------|-------------------------------|------------------|--------------------------------------------------|
| POST   | `/api/agent/ask`              | `{query}`        | `{answer, trace, iterations}`                    |
| GET    | `/api/agent/ask/stream`       | `?query=`        | SSE: `node_start` / `tool_call` / `tool_result` / `final_answer` / `done` / `error` |

### Pydantic-модели

```python
class AgentAskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)

class TraceEvent(BaseModel):
    type: Literal[
        "node_start", "tool_call", "tool_result",
        "final_answer", "error",
    ]
    timestamp: str           # ISO8601
    data: dict               # type-specific payload

class AgentAskResponse(BaseModel):
    answer: str
    trace: list[TraceEvent]
    iterations: int
```

### SSE event types

| Event           | Payload                                       | Когда                                           |
|-----------------|-----------------------------------------------|-------------------------------------------------|
| `node_start`    | `{node: "agent"\|"tools"}`                    | LangGraph входит в узел                         |
| `tool_call`     | `{name, args, id}`                            | LLM сгенерировала tool_call                     |
| `tool_result`   | `{name, args, id, result}`                    | tools-узел вернул результат                     |
| `final_answer`  | `{text}`                                      | LLM выдала ответ без tool_calls — граф к END    |
| `done`          | `{iterations}`                                | граф достиг END                                 |
| `error`         | `{code: "max_iter"\|"http_error"\|…, message}` | сбой                                            |

## Frontend

### Маршруты и компоненты

| Файл                                                | Что делает                                                                 |
|-----------------------------------------------------|----------------------------------------------------------------------------|
| `frontend/src/app/agent/page.tsx`                   | Маршрут `/agent`, рендерит `<AgentChat />`.                                |
| `frontend/src/components/agent/agent-chat.tsx`      | Список сообщений + composer (одна строка ввода). Без multi-chat.           |
| `frontend/src/components/agent/trace-timeline.tsx`  | Рендерит `trace[]` ассистент-сообщения как вертикальный список шагов.      |
| `frontend/src/components/agent/trace-step.tsx`      | Одна строка trace, collapsible. Типы: `thinking`, `tool_call`, `tool_result`, `final_answer`. |
| `frontend/src/lib/agent-types.ts`                   | `TraceEvent`, `AgentMessage`.                                              |
| `frontend/src/lib/agent-api.ts`                     | `ask(query)`, `buildStreamUrl(query)`.                                     |
| `frontend/src/stores/agent-ui.ts`                   | Zustand: текущий draft с накапливаемым trace, abort.                       |
| `frontend/src/components/sidebar/sidebar.tsx`       | Добавить вкладку «Agent» рядом с «Чаты».                                   |

### Поток данных

```
[UI] composer.send(query)
  → POST /api/agent/ask/stream (SSE открыт)
  ← node_start: agent
  ← tool_call: {name:"get_weather", args:{date:"2026-05-29"}, id:"t1"}
  ← node_start: tools
  ← tool_result: {id:"t1", result:{...}}
  ← node_start: agent
  ← tool_call: {name:"get_courts_availability", args:{...}, id:"t2"}
  ← node_start: tools
  ← tool_result: {id:"t2", result:{...}}
  ← node_start: agent
  ← final_answer: {text:"30 мая солнечно, корт свободен 18-20..."}
  ← done: {iterations: 3}

UI накапливает события в draft.trace[], рендерит timeline.
```

### Внешний вид trace в bubble

```
┌──────────────────────────────────────────────────┐
│ 🤖                                                │
│ [💭] Думаю над запросом…                          │
│ [🔧] get_weather(date="2026-05-29")              │
│   └ {temperature_c: 12, condition: "пасмурно"} ▾ │
│ [🔧] get_weather(date="2026-05-30")              │
│   └ {temperature_c: 22, condition: "ясно"}     ▾ │
│ [🔧] get_courts_availability(date="2026-05-30")  │
│   └ {summary: "Корт 2 свободен 18:00-20:00"}   ▾ │
│ [💬] Финальный ответ:                            │
│   30 мая солнечно, ~22°C. На корте 2 свободно   │
│   окно 18:00-20:00.                              │
└──────────────────────────────────────────────────┘
```

Каждый `[🔧]` collapsible — раскрывается на JSON request/response.

## Storage

Новая таблица для истории agent-чата:

```sql
CREATE TABLE IF NOT EXISTS agent_messages (
    id          BIGSERIAL PRIMARY KEY,
    role        TEXT NOT NULL,        -- "user" | "assistant"
    content     TEXT NOT NULL,
    trace       JSONB,                -- массив TraceEvent для assistant
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

В Spec 1 — **один глобальный поток сообщений**, без multi-chat. Это
снимает много вопросов про routing/UI и оставляет фокус на самом
агенте. Multi-chat — отдельная итерация.

## Зависимости

В `requirements.txt` добавить:

```
# --- LangGraph-агент ---
# Граф с узлами, состояниями и conditional edges.
langgraph>=0.2,<1.0

# ChatOpenAI-обёртка для LM Studio (он OpenAI-compat).
langchain-openai>=0.2,<1.0
# langchain-core придёт транзитивно (BaseMessage, ToolMessage, @tool).
```

`httpx` и `python-dotenv` уже есть.

## Конфиг (.env)

```
# --- Agent (Spec 1: skkrondo concierge) ---
SKKRONDO_BASE_URL=https://api.skkrondo.ru
```

`MAX_ITER` и тайм-ауты остаются константами в `agent/config.py` (для
Spec 1 в env не выносим).

## Известные риски и mitigations

1. **Gemma 4 e2b — нестабильный tool-calling.** Маленькие модели часто:
   забывают вызвать tool, возвращают кривой JSON tool_call, зацикливаются.
   - Mitigation A: усиленный system prompt с явными примерами того, когда
     надо звать tool.
   - Mitigation B: `MAX_ITER=10` — safety net против бесконечного loop'а.
   - Mitigation C: если на практике плохо — апгрейд на Gemma 4 e4b или
     больше (на стороне пользователя в LM Studio). Само приложение
     не меняется, только `CHAT_MODEL` в `.env`.

2. **Skkrondo API нестабилен / медленный.** Возможны 5xx / timeouts.
   - Mitigation: HTTP-клиент с тайм-аутом 10s. Tool возвращает `{error: …}`,
     LLM сама решит retry или сообщить пользователю.

3. **Параллельные tool_calls в одном turn'е.** LangGraph умеет, но Gemma
   скорее всего сгенерирует по одному. Это нормально — будет 2N
   итераций вместо N, но логически корректно.

4. **Streaming токенов финального ответа.** Сложнее с tools — отложим
   до отдельной итерации. В Spec 1 финальный текст приходит целиком
   через `final_answer` event.

## Тестирование

Manual через curl + браузер. Сценарии:

1. **Weather-only**: «Какая погода 1 июня?» → trace с одним `get_weather` →
   осмысленный ответ с температурой.
2. **Courts-only**: «Что свободно 1 июня?» → один `get_courts_availability` →
   ответ со слотами.
3. **Главный кейс — связка**: «Найди солнечный день на следующей неделе
   со свободным кортом» → trace с ~14 tool_calls (7 дат × 2 tool'а), ответ
   с конкретными датами/окнами.
4. **Out-of-scope**: «Что в кино идёт?» → LLM не зовёт tools, отвечает
   что вне зоны.
5. **Tool-error**: временно `SKKRONDO_BASE_URL=http://localhost:1` →
   tool возвращает error → trace показывает ошибку → LLM сообщает что
   данные недоступны.
6. **Max-iterations**: придумать запрос на который LLM зациклится
   (например «продолжай искать пока не найдёшь идеальный день») →
   видим `error: max_iter_exceeded`, корректное завершение.

## Out of scope (для других spec'ов)

- **Booking-tools** (POST endpoints). Опасно без auth.
- **RAG-tool** — RAG как третий tool для FAQ/регламента клуба.
- **Multi-chat** (как в `/`). Один глобальный поток в Spec 1.
- **Token-streaming** финального ответа.
- **Параллельные tool_calls** в одном turn'е (LangGraph умеет, не форсим).
- **Полноценный graph-визуализатор** (SVG/canvas с подсветкой текущего
  узла). Trace timeline в Spec 1 — линейный список. Граф-view —
  отдельная итерация.
- **Tool-retry policy** на сетевых ошибках. Пока — error в LLM-context.

## Backwards compatibility

- Существующие RAG-маршруты `/api/*` и `/chat/*` не меняются.
- Существующие таблицы (`chunks`, `chats`, `messages`) не трогаются.
- `.env.example` дополняется одной переменной `SKKRONDO_BASE_URL`.
- Sidebar получает вторую вкладку — текущий RAG-чат остаётся на «Чаты».

## Следующий шаг

После согласования spec'а — переход к writing-plans skill для
пошагового плана реализации.
