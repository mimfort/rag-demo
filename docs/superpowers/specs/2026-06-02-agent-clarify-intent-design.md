# Дизайн: шаг подтверждения намерения (clarify) у агента

**Дата:** 2026-06-02
**Статус:** одобрено к планированию

## Проблема и цель

Сейчас агент (LangGraph, узлы `agent ⇄ tools`) выполняет запрос за один
проход: `HumanMessage → граф → финальный ответ`, без остановок. Пользователь
не видит, *как именно* агент понял запрос: «какая погода сегодня?» молча
превращается в `get_weather("2026-06-02")`.

**Цель:** первым шагом каждого turn'а агент перефразирует запрос пользователя
(резолвит относительные даты «сегодня»/«на выходных» → конкретные ISO-даты,
проясняет размытые формулировки) и показывает: *«Вы имели в виду: …? Да/Нет»*.
Только после подтверждения агент идёт работать.

## Решения (зафиксированы при брейншторме)

1. **Когда спрашивать:** всегда, первым шагом каждого turn'а.
2. **Поведение на «нет» + уточнение:** цикл — агент заново перефразирует с
   учётом уточнения и спрашивает снова, пока не «да».
3. **Реализация:** LangGraph `interrupt()` внутри графа (human-in-the-loop) +
   checkpointer + `thread_id` + resume-эндпоинт.

## Архитектура

### Форма графа

Было:

```
START → agent ⇄ tools → END
```

Стало:

```
START → interpret → confirm ──(да)──→ agent ⇄ tools → END
            ↑                  │
            └──────(нет)───────┘
```

### Узлы

Два новых узла **вместо одного** — намеренное разделение, чтобы обойти
семантику LangGraph «узел с `interrupt()` перезапускается с начала при resume»:

- **`interpret`** — один LLM-вызов. Берёт исходный запрос пользователя (+
  `correction` из state, если уже был круг «нет»), резолвит относительные даты
  по `today`, возвращает строку-перефразировку в `state["interpretation"]`.
  LLM зовётся **ровно один раз за круг**.
- **`confirm`** — вызывает `answer = interrupt({...})` → пауза. **Без**
  LLM-вызова, поэтому при resume перезапускается дёшево. По ответу:
  - `confirmed=true` → `state["effective_query"] = interpretation`; conditional
    edge → `agent`;
  - `confirmed=false` → `state["correction"] = <уточнение>`; conditional edge
    → `interpret` (новый круг).

**Защита от зацикливания:** `MAX_CLARIFY_ROUNDS` (значение: 5). После
превышения `confirm` принудительно проставляет `effective_query` и пропускает
в `agent` по последней интерпретации.

**`agent_node`:** при наличии `effective_query` добавляет его в контекст LLM
как подтверждённую формулировку запроса (исходный `HumanMessage` остаётся в
истории — `effective_query` его дополняет, а не заменяет).

### State (`agent/state.py`)

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    interpretation: str          # последняя предложенная перефразировка
    effective_query: str         # подтверждённая формулировка (для agent_node)
    correction: str | None       # последнее уточнение пользователя
    clarify_rounds: int          # счётчик кругов (для MAX_CLARIFY_ROUNDS)
```

### Checkpointer и thread_id

- **Checkpointer:** `MemorySaver` (`langgraph.checkpoint.memory`) — in-process,
  без схемы БД. Достаточно для дев-демо на одном процессе.
  - *Известное ограничение:* чекпоинты живут в памяти процесса; при `--reload`
    uvicorn теряются (как и любое in-memory состояние сейчас).
  - *Будущий апгрейд (вне скоупа):* `PostgresSaver` — в проекте уже есть
    Postgres-соединение; позволит переживать рестарты.
- **Граф компилируется с checkpointer'ом:** `build_graph()` →
  `.compile(checkpointer=MemorySaver())`. Остаётся module-singleton в
  `runner.py`.
- **thread_id:** один на turn. Генерирует **бэкенд** на старте (`uuid4`),
  отдаёт клиенту в первом `clarify`-событии. Клиент возвращает его при resume.
  Все вызовы графа — с `config={"configurable": {"thread_id": tid}}`.

## Бэкенд: runner и эндпоинты

### Новое SSE-событие

`clarify` с данными:

```json
{ "thread_id": "…", "interpretation": "…", "original": "…", "round": 1 }
```

Когда `confirm` вызывает `interrupt()`, `graph.astream(stream_mode="updates")`
отдаёт спец-апдейт `{"__interrupt__": (...)}`. Runner ловит его, эмитит SSE
`clarify` и штатно завершает стрим (граф «на паузе», сохранён в checkpointer).

### `agent/runner.py`

Общий `_astream_events(thread_id, *, query=None, resume=None)`:

- если `resume` передан → `astream(Command(resume={"confirmed", "correction"}),
  config={thread_id}, stream_mode="updates")`;
- иначе → `astream({"messages": [HumanMessage(query)]}, config={thread_id},
  stream_mode="updates")`.

Разбор апдейтов в одном месте: `__interrupt__` → событие `clarify`; всё
остальное — как сейчас (`node_start / tool_call / tool_result / final_answer /
done / error`).

### Эндпоинты (`web/server.py`)

Оба **GET + SSE** (чтобы переиспользовать браузерный `EventSource`):

| Эндпоинт | Назначение |
|---|---|
| `GET /api/agent/ask/stream?query=…` | старт turn'а. Бэк генерит `thread_id`, стримит до первого `clarify`, затем стрим закрывается. |
| `GET /api/agent/resume/stream?thread_id=…&confirmed=…&correction=…` | возобновление. Либо снова `clarify` (круг «нет»), либо проход `agent ⇄ tools` → `final_answer` → `done`. |

Старый путь `/api/agent/ask/stream` сохраняет имя; меняется лишь то, что его
первый ответ теперь — `clarify`, а не сразу работа агента.

### Сохранение в БД и trace

Clarify-круги и финальный ответ размазаны по нескольким HTTP-запросам одного
turn'а. Решение:

- in-memory дикт `thread_id → accumulated_trace`, дополняется на каждом событии
  всех запросов turn'а (парно с `MemorySaver` — тоже in-memory);
- user-сообщение пишется в `agent_messages` на старте (как сейчас);
- на `done` сохраняется assistant-сообщение с **полным** накопленным trace,
  запись из дикта удаляется.

`MAX_ITER`-гард остаётся для цикла `agent ⇄ tools`; у clarify-цикла свой гард
`MAX_CLARIFY_ROUNDS` (в графе).

## Фронтенд

### Типы (`agent-types.ts`)

- В `AgentEventType` добавить `"clarify"`.
- Тип данных события: `{ thread_id: string; interpretation: string; original:
  string; round: number }`.

### API-клиент (`agent-api.ts`)

- Добавить `buildResumeUrl({ threadId, confirmed, correction })` →
  `/api/agent/resume/stream?…` (по аналогии с `buildStreamUrl`).

### Store (`agent-ui.ts`)

- В `draft` добавить `pendingClarify: { threadId, interpretation, original,
  round } | null`.
- При событии `clarify` — выставить `pendingClarify`; turn **не** финализируется
  (`draft` остаётся, `finished=false`).

### UI (`agent-chat.tsx` + новый `clarify-prompt.tsx`)

- Логику `EventSource` вынести в переиспользуемую `openStream(url, handlers)` —
  зовётся и на старте, и на resume (общий набор слушателей, включая `clarify`).
- При `pendingClarify` под пузырём «Думаю…» рисуется карточка:

  ```
  🤔 Вы имели в виду:
     «Какая погода будет 2026-06-02?»

     [ Да, верно ]   [ Нет ]
     (по «Нет» появляется поле: что вы имели в виду)
  ```

  - **«Да, верно»** → `openStream(buildResumeUrl({threadId, confirmed:true}))`,
    карточка убирается, агент идёт работать.
  - **«Нет»** → textarea; на отправку →
    `openStream(buildResumeUrl({threadId, confirmed:false, correction}))` →
    приходит новый `clarify` → карточка обновляется.

- Каждое `clarify`-событие также добавляется в `trace` через `appendTrace`,
  чтобы в `TraceTimeline` остался след «о чём договаривались». В `trace-step.tsx`
  добавить отрисовку шага типа `clarify`.
- `done / error` — как сейчас (финализация draft, инвалидация `messages`).

## Затрагиваемые файлы

**Бэкенд:**
- `agent/state.py` — новые поля state.
- `agent/graph.py` — узлы `interpret` / `confirm`, conditional edges, компиляция
  с `MemorySaver`, промпт интерпретатора.
- `agent/runner.py` — `_astream_events` с поддержкой `resume`/`thread_id`,
  обработка `__interrupt__`, накопление trace по `thread_id`.
- `agent/config.py` — `MAX_CLARIFY_ROUNDS`.
- `web/server.py` — генерация `thread_id` на старте, новый эндпоинт
  `/api/agent/resume/stream`, событие `clarify`, сохранение trace на `done`.

**Фронтенд:**
- `frontend/src/lib/agent-types.ts` — тип события `clarify`.
- `frontend/src/lib/agent-api.ts` — `buildResumeUrl`.
- `frontend/src/stores/agent-ui.ts` — `pendingClarify`.
- `frontend/src/components/agent/agent-chat.tsx` — `openStream`, рендер карточки.
- `frontend/src/components/agent/clarify-prompt.tsx` — новый компонент.
- `frontend/src/components/agent/trace-step.tsx` — отрисовка шага `clarify`.

## Вне скоупа

- `PostgresSaver` вместо `MemorySaver` (durable-чекпоинты через рестарты).
- Подтверждение «только при неоднозначности» (выбрано «всегда»).
- Тайм-аут авто-подтверждения по молчанию.

## Тестирование

- **Бэкенд (unit):** граф с `MemorySaver` — старт даёт `__interrupt__` с
  интерпретацией; `Command(resume={confirmed:true})` доходит до `agent`;
  `Command(resume={confirmed:false, correction})` даёт новый `__interrupt__`;
  превышение `MAX_CLARIFY_ROUNDS` пропускает в `agent`.
- **Бэкенд (интеграция):** `GET /ask/stream` отдаёт событие `clarify` с
  `thread_id`; `GET /resume/stream?...&confirmed=true` доводит до `done`; trace в
  `agent_messages` содержит clarify-круги и финальный ответ.
- **Фронтенд:** при `clarify` рисуется карточка; «Да» запускает работу агента;
  «Нет» + уточнение шлёт resume и обновляет карточку.
</content>
</invoke>
