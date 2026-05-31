# LangGraph Agent (Sport-concierge for SKK Rondo) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Реализовать ReAct-агента для СКК «Рондо», который через LangGraph оркестрирует два tools (`get_weather`, `get_courts_availability`) над публичным API `api.skkrondo.ru`, и UI с trace-timeline на `/agent`.

**Architecture:** LangGraph `StateGraph` с двумя узлами (`agent` LLM-узел + `tools` `ToolNode`) и conditional edge через `tools_condition`. LLM = существующая Gemma-4 e2b в LM Studio через `langchain-openai`. Tools — async httpx-обёртки. Backend: 2 endpoint'а под `/api/agent/*`, новая таблица `agent_messages`. Frontend: вкладка «Agent» в sidebar + маршрут `/agent` с collapsible trace-шагами.

**Tech Stack:** Python 3.13 + FastAPI (async endpoints) + langgraph + langchain-openai + httpx + psycopg v3; Next.js 15 + TanStack Query + Zustand + shadcn/ui.

**Spec:** `docs/superpowers/specs/2026-05-28-langgraph-agent-design.md`

---

## File map

**Backend:**
- Create: `agent/__init__.py`, `agent/config.py`, `agent/state.py`, `agent/tools.py`, `agent/llm.py`, `agent/graph.py`, `agent/runner.py`
- Create: `db/init_agent.sql` (таблица `agent_messages`)
- Modify: `requirements.txt` (+`langgraph`, `langchain-openai`)
- Modify: `.env.example` (+`SKKRONDO_BASE_URL`)
- Modify: `web/server.py` (новая TG-aналогичная секция «Agent» внизу — Pydantic-модели, `_agent_store`/`_agent_graph` singletons, 3 ручки: `/ask`, `/ask/stream`, `/messages`)

**Frontend:**
- Create: `frontend/src/lib/agent-types.ts`, `frontend/src/lib/agent-api.ts`
- Create: `frontend/src/stores/agent-ui.ts`
- Create: `frontend/src/components/agent/agent-chat.tsx`, `trace-timeline.tsx`, `trace-step.tsx`
- Create: `frontend/src/app/agent/page.tsx`
- Create (если нет): `frontend/src/components/ui/tabs.tsx` (shadcn-обёртка вокруг radix-tabs)
- Modify: `frontend/src/components/sidebar/sidebar.tsx` (две вкладки: «Чаты» и «Agent»)

---

## Pre-flight

- [ ] **Step 0a: Убедиться, что бэкенд+фронт+postgres живы**

Run:
```bash
docker ps --format "{{.Names}}\t{{.Status}}" | head
curl -s http://localhost:8000/api/health
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:3000
```

Expected: `rag_postgres` в `Up healthy`, backend `{"ok":true,"chunks_in_db":...}`, frontend `200`.

Если что-то не работает: `docker compose up -d`, `uvicorn web.server:app --host 0.0.0.0 --port 8000 &`, `cd frontend && npm run dev &`.

---

### Task 1: Зависимости и env

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`

- [ ] **Step 1: Добавить две зависимости в `requirements.txt`**

Дописать в конец:

```
# --- LangGraph-агент (Spec 1: skkrondo concierge) ---
# Граф с узлами, состояниями и conditional edges.
langgraph>=0.2,<1.0

# ChatOpenAI-обёртка для LM Studio (он OpenAI-compat). langchain-core
# придёт транзитивно — даст BaseMessage, ToolMessage, @tool decorator.
langchain-openai>=0.2,<1.0
```

- [ ] **Step 2: Установить**

Run:
```bash
.venv/bin/pip install -r requirements.txt
```

Expected: `Successfully installed langgraph-... langchain-openai-... langchain-core-...` (или «already satisfied»). Без ошибок резолюции.

- [ ] **Step 3: Добавить переменную в `.env.example`**

В конец `.env.example`:

```
# --- Agent (Spec 1: skkrondo concierge) ---
SKKRONDO_BASE_URL=https://api.skkrondo.ru
```

- [ ] **Step 4: Дописать ту же переменную в свой `.env`**

Run:
```bash
echo "" >> .env
echo "# --- Agent (Spec 1: skkrondo concierge) ---" >> .env
echo "SKKRONDO_BASE_URL=https://api.skkrondo.ru" >> .env
tail -3 .env
```

Expected: видишь три новые строки.

- [ ] **Step 5: Smoke-тест импорта**

Run:
```bash
.venv/bin/python -c "from langgraph.graph import StateGraph, END; from langgraph.prebuilt import ToolNode, tools_condition; from langchain_openai import ChatOpenAI; from langchain_core.tools import tool; print('ok')"
```

Expected: `ok` без ошибок.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example
git commit -m "$(cat <<'EOF'
feat(agent): add langgraph + langchain-openai deps and SKKRONDO_BASE_URL env

Заготовка для ReAct-агента: skkrondo concierge. ChatOpenAI работает с
LM Studio через OpenAI-compat API; langgraph даёт StateGraph + готовый
ToolNode и tools_condition edge-helper.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: DB schema для agent

**Files:**
- Create: `db/init_agent.sql`

- [ ] **Step 1: Создать `db/init_agent.sql`**

```sql
-- =============================================================================
-- init_agent.sql — таблица для LangGraph-агента (Spec 1).
-- Выполняется автоматически Docker'ом при первом старте контейнера
-- (docker-entrypoint-initdb.d), либо вручную psql'ом для running БД.
-- =============================================================================

CREATE TABLE IF NOT EXISTS agent_messages (
    id          BIGSERIAL PRIMARY KEY,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    trace       JSONB,                -- массив TraceEvent для assistant; для user — NULL
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_messages_created_at_idx
    ON agent_messages (created_at);
```

- [ ] **Step 2: Применить к running БД**

```bash
docker exec -i rag_postgres psql -U rag -d rag < db/init_agent.sql
```

Expected: `CREATE TABLE` + `CREATE INDEX`, без ошибок.

- [ ] **Step 3: Проверить**

```bash
docker exec rag_postgres psql -U rag -d rag -c "\d agent_messages"
```

Expected: видишь 5 столбцов (id BIGSERIAL, role TEXT, content TEXT, trace JSONB, created_at TIMESTAMPTZ).

- [ ] **Step 4: Commit**

```bash
git add db/init_agent.sql
git commit -m "$(cat <<'EOF'
feat(agent): add DDL for agent_messages table

Один глобальный поток сообщений (без multi-chat в Spec 1). Поле trace
(JSONB) хранит массив TraceEvent для assistant-сообщений — даёт
восстановление шагов trace после перезагрузки страницы.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `agent/config.py` и `agent/state.py`

**Files:**
- Create: `agent/__init__.py`
- Create: `agent/config.py`
- Create: `agent/state.py`

- [ ] **Step 1: Создать `agent/__init__.py`**

```python
"""LangGraph-агент (Spec 1: skkrondo concierge) — ReAct поверх локальной LLM."""
```

- [ ] **Step 2: Создать `agent/config.py`**

```python
"""
Настройки агент-модуля. Большинство значений живёт в .env (через
python-dotenv в rag/config.py), небольшие технические константы — здесь.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


# Защитный лимит итераций ReAct-loop (agent → tools → agent → ...).
# При превышении runner emit'нёт error и завершит граф.
MAX_ITER: int = 10

# Таймаут httpx-вызовов к skkrondo API (одного запроса).
HTTP_TIMEOUT_SEC: float = 10.0


@dataclass(frozen=True)
class AgentSettings:
    skkrondo_base_url: str

    @property
    def is_configured(self) -> bool:
        return bool(self.skkrondo_base_url)


def load_agent_settings() -> AgentSettings:
    return AgentSettings(
        skkrondo_base_url=(
            os.getenv("SKKRONDO_BASE_URL") or "https://api.skkrondo.ru"
        ).rstrip("/"),
    )


agent_settings = load_agent_settings()
```

- [ ] **Step 3: Создать `agent/state.py`**

```python
"""
State-схема для LangGraph-агента. Только одно поле — `messages` — это
канонический ReAct-state в LangGraph (вся история turn'а лежит в нём).
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # `add_messages` — встроенный reducer LangGraph: при обновлении state
    # из узла новые сообщения АППЕНДЯТСЯ к существующим, а не заменяют.
    # Это правильное поведение для ReAct: каждый узел добавляет 1+ message.
    messages: Annotated[list[BaseMessage], add_messages]
```

- [ ] **Step 4: Smoke-импорт**

```bash
.venv/bin/python -c "from agent.config import agent_settings, MAX_ITER; from agent.state import AgentState; print('base_url:', agent_settings.skkrondo_base_url, '| max_iter:', MAX_ITER)"
```

Expected: `base_url: https://api.skkrondo.ru | max_iter: 10`.

- [ ] **Step 5: Commit**

```bash
git add agent/__init__.py agent/config.py agent/state.py
git commit -m "$(cat <<'EOF'
feat(agent): config (skkrondo URL, MAX_ITER) and state schema

AgentState — каноничная LangGraph-схема для ReAct: единственное поле
messages с add_messages-reducer'ом (новые сообщения append'ятся к
существующим, а не заменяют).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `agent/tools.py`

**Files:**
- Create: `agent/tools.py`

- [ ] **Step 1: Написать `agent/tools.py`**

```python
"""
Tools для LangGraph-агента. Каждая обёртка:
  1. Дёргает один GET-эндпоинт skkrondo API через httpx.
  2. Преобразует сырой JSON в компактный человекочитаемый dict —
     это снижает нагрузку на контекст LLM (Gemma 4 e2b маленькая).
  3. На ошибки сети/HTTP возвращает {"error": "..."} вместо exception —
     LLM сам решит retry или сообщить пользователю.

@tool — это langchain-core декоратор, который превращает функцию в
ToolNode-совместимый callable. Сигнатуру и docstring decorator
автоматически конвертирует в JSON-schema description для LLM.
"""

from __future__ import annotations

import httpx
from langchain_core.tools import tool

from agent.config import HTTP_TIMEOUT_SEC, agent_settings


# Условия в weather-API, которые мы считаем «солнечно». На реальной
# структуре ответа skkrondo может быть лучше уточнить (см. /weather/weather/{date}).
_SUNNY_KEYWORDS = ("ясно", "малооблачно", "солнечно", "clear", "sunny")


def _classify_sunny(raw: dict) -> bool:
    """Эвристика по полю condition/description — солнечно или нет.
    Финальная интерпретация на LLM — это просто подсказка."""
    text = (
        str(raw.get("condition") or raw.get("description") or "")
    ).lower()
    return any(kw in text for kw in _SUNNY_KEYWORDS)


def _summarize_courts(raw: list) -> str:
    """Сжимаем массив броней в одну строку — чтобы не раздувать LLM-контекст."""
    if not raw:
        return "Все корты свободны."
    by_court: dict[int, list[str]] = {}
    for r in raw:
        court = r.get("court_id") or r.get("court", "?")
        slot = f"{r.get('start_time', '?')}-{r.get('end_time', '?')}"
        by_court.setdefault(court, []).append(slot)
    parts = [
        f"корт {c}: занят {', '.join(slots)}"
        for c, slots in sorted(by_court.items(), key=lambda x: str(x[0]))
    ]
    return "; ".join(parts)


@tool
async def get_weather(date: str) -> dict:
    """Получить прогноз погоды на конкретную дату в формате YYYY-MM-DD.

    Возвращает: {date, temperature_c, condition, sunny: bool, raw}
    либо {error: "..."} при сетевой/HTTP ошибке.
    """
    url = f"{agent_settings.skkrondo_base_url}/weather/weather/{date}"
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
            r = await client.get(url)
            r.raise_for_status()
            raw = r.json()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "date": date}
    return {
        "date": date,
        "temperature_c": raw.get("temperature") or raw.get("temp"),
        "condition": raw.get("condition") or raw.get("description"),
        "sunny": _classify_sunny(raw),
        "raw": raw,
    }


@tool
async def get_courts_availability(date: str) -> dict:
    """Получить занятые слоты кортов на конкретную дату YYYY-MM-DD.

    Возвращает: {date, reservations: [...], summary: "текст"} либо
    {error: "..."} при сетевой/HTTP ошибке.
    """
    url = f"{agent_settings.skkrondo_base_url}/court_reservations/all/{date}"
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
            r = await client.get(url)
            r.raise_for_status()
            raw = r.json()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "date": date}
    return {
        "date": date,
        "reservations": raw,
        "summary": _summarize_courts(raw),
    }


TOOLS = [get_weather, get_courts_availability]
```

- [ ] **Step 2: Live smoke-тест каждого tool'а**

```bash
.venv/bin/python -c "
import asyncio
from datetime import date, timedelta
from agent.tools import get_weather, get_courts_availability

async def main():
    today = date.today().isoformat()
    print('=== weather ===')
    r = await get_weather.ainvoke({'date': today})
    print(r)
    print()
    print('=== courts ===')
    r = await get_courts_availability.ainvoke({'date': today})
    print({k: v for k, v in r.items() if k != 'reservations'})

asyncio.run(main())
"
```

Expected: каждый tool возвращает dict — либо с реальными данными, либо с `{"error": ...}` если skkrondo API недоступен. БЕЗ необработанных исключений.

- [ ] **Step 3: Commit**

```bash
git add agent/tools.py
git commit -m "$(cat <<'EOF'
feat(agent): add get_weather and get_courts_availability tools

Каждая обёртка — async httpx + парсинг в компактный dict (summary для
courts, sunny-heuristic для weather). Ошибки сети/HTTP не пробрасываются,
а возвращаются как {error: "..."} — LLM сам решит retry или сказать
пользователю.

@tool decorator из langchain-core делает функции совместимыми с
ToolNode (LangGraph) и автоматически генерирует JSON-schema из
сигнатуры/docstring для LLM.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `agent/llm.py`

**Files:**
- Create: `agent/llm.py`

- [ ] **Step 1: Написать `agent/llm.py`**

```python
"""
Фабрика LLM. ChatOpenAI работает с LM Studio: тот OpenAI-совместим,
просто base_url другой.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from rag.config import settings as rag_settings


def make_llm() -> ChatOpenAI:
    """
    Возвращает ChatOpenAI настроенную на локальную LM Studio.

    temperature=0.2 — мы хотим стабильный tool-calling, не креатив.
    max_retries=1 — обычно LM Studio либо отвечает, либо нет; retry не
    спасёт.
    """
    return ChatOpenAI(
        model=rag_settings.chat_model,
        base_url=rag_settings.lm_studio_base_url,
        api_key=rag_settings.lm_studio_api_key,
        temperature=0.2,
        max_retries=1,
    )
```

- [ ] **Step 2: Smoke-импорт**

```bash
.venv/bin/python -c "from agent.llm import make_llm; llm = make_llm(); print('model:', llm.model_name, '| base_url:', llm.openai_api_base)"
```

Expected: видишь имя модели и URL LM Studio.

- [ ] **Step 3: Commit**

```bash
git add agent/llm.py
git commit -m "$(cat <<'EOF'
feat(agent): ChatOpenAI factory for LM Studio

ChatOpenAI работает с LM Studio как с обычным OpenAI-API: base_url + api_key
из rag/config.py settings. temperature=0.2 для стабильного tool-calling.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `agent/graph.py` — сборка StateGraph

**Files:**
- Create: `agent/graph.py`

- [ ] **Step 1: Написать `agent/graph.py`**

```python
"""
Сборка LangGraph: ReAct-цикл agent ⇄ tools.

Структура:
    START
      ↓
    agent  ─── tool_calls? ──→ tools
      ↑                          │
      └──────────────────────────┘
      (если нет tool_calls → END)
"""

from __future__ import annotations

from datetime import date

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.llm import make_llm
from agent.state import AgentState
from agent.tools import TOOLS


SYSTEM_PROMPT_TEMPLATE = """Ты — помощник по СКК «Рондо». Ты можешь вызывать tools чтобы получить погоду и список забронированных кортов на конкретные даты.

Сегодня: {today}.

Если пользователь спрашивает про «свободный день», «следующую неделю», «погоду» — определи диапазон дат и проверь их tool'ами. Когда соберёшь достаточно данных — отвечай по-русски, упоминай конкретные даты и часы.

Не вызывай один и тот же tool с одинаковыми аргументами повторно. Когда ответ можно дать — отвечай без новых tool_call'ов."""


def _build_system_message() -> SystemMessage:
    return SystemMessage(
        SYSTEM_PROMPT_TEMPLATE.format(today=date.today().isoformat())
    )


def build_graph():
    """Создаёт скомпилированный граф. Вызывается один раз на процесс
    (heavy — bind_tools шлёт каждый раз tools-schema'у в LLM на инвоке).
    """
    llm_with_tools = make_llm().bind_tools(TOOLS)

    async def agent_node(state: AgentState) -> dict:
        """Вызов LLM. Если в state нет system-message — добавляем
        в начало (даты живые, не вшиты в build)."""
        messages = list(state["messages"])
        if not messages or messages[0].type != "system":
            messages = [_build_system_message()] + messages
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    # tools_condition возвращает "tools" если у последнего сообщения
    # есть .tool_calls, иначе END.
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()
```

- [ ] **Step 2: Smoke-build**

```bash
.venv/bin/python -c "from agent.graph import build_graph; g = build_graph(); print('nodes:', list(g.get_graph().nodes), '| compiled:', g is not None)"
```

Expected: `nodes: ['__start__', 'agent', 'tools', '__end__'] | compiled: True`.

- [ ] **Step 3: Commit**

```bash
git add agent/graph.py
git commit -m "$(cat <<'EOF'
feat(agent): assemble LangGraph (agent ⇄ tools)

Минимальный ReAct: agent-узел вызывает LLM с bound tools, tools-узел
исполняет tool_calls, conditional edge через tools_condition зацикливает
пока LLM не выдаст ответ без tool_calls.

System-message с актуальной датой инжектируется в начало messages на
каждом вызове agent-узла (не вшито в build_graph, чтобы дата была живая).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `agent/runner.py` — event-stream wrapper

**Files:**
- Create: `agent/runner.py`

- [ ] **Step 1: Написать `agent/runner.py`**

```python
"""
Runner — обёртка над graph.astream(stream_mode="updates"), которая
конвертирует LangGraph-обновления в наши SSE-events.

Event types (см. spec):
  - node_start  {node}             — вошли в узел
  - tool_call   {name, args, id}   — agent сгенерировал tool_call
  - tool_result {name, args, id, result} — tools-узел вернул результат
  - final_answer {text}            — agent выдал ответ без tool_calls
  - done        {iterations}       — граф достиг END
  - error       {code, message}    — exception или превышен MAX_ITER
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.config import MAX_ITER
from agent.graph import build_graph


@dataclass
class AgentRun:
    """Результат полного прогона графа (non-streaming режим). Trace —
    плоский список dict'ов (тех же что в SSE-events)."""
    answer: str
    trace: list[dict] = field(default_factory=list)
    iterations: int = 0


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _event(type_: str, data: dict) -> dict:
    return {"type": type_, "timestamp": _now_iso(), "data": data}


async def _astream_events(query: str) -> AsyncIterator[dict]:
    """Внутренний async-генератор: yield'ит dict-events по мере
    выполнения графа. НЕ форматирует SSE — это делает Web-слой."""
    graph = build_graph()
    iterations = 0

    try:
        # stream_mode="updates" даёт {node_name: state_delta} после каждого узла.
        async for update in graph.astream(
            {"messages": [HumanMessage(query)]},
            stream_mode="updates",
        ):
            iterations += 1
            if iterations > MAX_ITER:
                yield _event("error", {
                    "code": "max_iter",
                    "message": f"Превышен лимит итераций ({MAX_ITER})",
                })
                return

            for node_name, delta in update.items():
                yield _event("node_start", {"node": node_name})
                new_messages = delta.get("messages") or []
                for msg in new_messages:
                    if isinstance(msg, AIMessage):
                        tool_calls = msg.tool_calls or []
                        if tool_calls:
                            for tc in tool_calls:
                                yield _event("tool_call", {
                                    "name": tc["name"],
                                    "args": tc["args"],
                                    "id": tc["id"],
                                })
                        else:
                            yield _event("final_answer", {
                                "text": msg.content or "",
                            })
                    elif isinstance(msg, ToolMessage):
                        # ToolMessage.content — строка (JSON-сериализованный
                        # результат tool'а). Пытаемся распарсить обратно
                        # для красивого отображения; если не парсится —
                        # отдаём как есть.
                        result: object = msg.content
                        try:
                            result = json.loads(msg.content)
                        except (json.JSONDecodeError, TypeError):
                            pass
                        yield _event("tool_result", {
                            "name": msg.name or "",
                            "id": msg.tool_call_id or "",
                            "result": result,
                        })

        yield _event("done", {"iterations": iterations})

    except Exception as exc:
        yield _event("error", {
            "code": type(exc).__name__,
            "message": str(exc),
        })


async def run_collect(query: str) -> AgentRun:
    """Прогнать граф без стриминга — собрать trace и финальный ответ.
    Используется в `POST /api/agent/ask`."""
    trace: list[dict] = []
    answer = ""
    iterations = 0
    async for event in _astream_events(query):
        trace.append(event)
        if event["type"] == "final_answer":
            answer = event["data"]["text"]
        elif event["type"] == "done":
            iterations = event["data"]["iterations"]
        elif event["type"] == "error":
            answer = answer or f"Ошибка: {event['data']['message']}"
    return AgentRun(answer=answer, trace=trace, iterations=iterations)


async def run_stream(query: str) -> AsyncIterator[dict]:
    """Прогнать граф стримом — yield'ить events по мере появления.
    Используется в `GET /api/agent/ask/stream`."""
    async for event in _astream_events(query):
        yield event
```

- [ ] **Step 2: Smoke — прогон коллектора**

```bash
.venv/bin/python -c "
import asyncio
from agent.runner import run_collect

async def main():
    result = await run_collect('Какая сегодня погода?')
    print('iterations:', result.iterations)
    print('events:', len(result.trace))
    print('event types:', [e['type'] for e in result.trace])
    print('answer:', result.answer[:200])

asyncio.run(main())
" 2>&1 | tail -20
```

Expected: видим события `node_start → tool_call → node_start → tool_result → node_start → final_answer → done` (примерно). `answer` непустой.

ВАЖНО: если LM Studio не отвечает или Gemma не справилась с tool-calling — получишь `error` событие. Это **тоже валидный результат теста**: видим, что run_collect не падает, а корректно завершается с trace + error в нём.

- [ ] **Step 3: Commit**

```bash
git add agent/runner.py
git commit -m "$(cat <<'EOF'
feat(agent): runner.py — graph.astream → typed events

run_stream/run_collect — публичные entry-point'ы. Внутри _astream_events
читает LangGraph-обновления и эмитит 6 типов событий (node_start,
tool_call, tool_result, final_answer, done, error). Safety net MAX_ITER
эмитит error event и корректно завершается без exception'а.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Backend storage — `agent_store` поверх `agent_messages`

**Files:**
- Modify: `web/server.py` (новая секция в конце файла + один helper в startup'е)

- [ ] **Step 1: Импорты в `web/server.py`**

В верхней части файла, рядом с импортами `rag.*`, добавить:

```python
from agent.runner import run_collect, run_stream
```

(`run_stream` понадобится в Task 9 — пусть будет импорт сразу.)

- [ ] **Step 2: Добавить TG-аналогичную секцию в конце файла**

В `web/server.py` НИЖЕ всех существующих эндпоинтов и блока `import uuid` (если он есть) добавить:

```python
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


@app.post("/api/agent/ask", response_model=AgentAskResponse)
async def agent_ask(req: AgentAskRequest) -> AgentAskResponse:
    """Non-streaming прогон агента. Сохраняет user+assistant пару в БД."""
    _save_agent_message("user", req.query, None)
    result = await run_collect(req.query)
    _save_agent_message("assistant", result.answer, result.trace)
    return AgentAskResponse(
        answer=result.answer,
        trace=[TraceEvent(**e) for e in result.trace],
        iterations=result.iterations,
    )


@app.get("/api/agent/messages", response_model=list[AgentMessageOut])
def agent_messages(limit: int = 200) -> list[AgentMessageOut]:
    return _list_agent_messages(limit=limit)
```

- [ ] **Step 3: Перезапустить uvicorn**

```bash
pkill -f "uvicorn web.server" 2>/dev/null; sleep 1
.venv/bin/uvicorn web.server:app --host 0.0.0.0 --port 8000
```
(Запускай через Bash run_in_background=true.)

```bash
until curl -s http://localhost:8000/api/health 2>/dev/null | grep -q '"ok":true'; do sleep 0.5; done; echo "ready"
```

- [ ] **Step 4: Smoke — пустая история и тестовый ask**

```bash
echo "=== empty history ==="
curl -s http://localhost:8000/api/agent/messages | python3 -m json.tool

echo "=== ask ==="
curl -s -X POST http://localhost:8000/api/agent/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"Какая сегодня погода?"}' --max-time 120 \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('answer:', d['answer'][:200]); print('iterations:', d['iterations']); print('trace events:', [e['type'] for e in d['trace']])"

echo "=== history after ==="
curl -s http://localhost:8000/api/agent/messages | python3 -c "import json,sys; d=json.load(sys.stdin); print('messages:', len(d)); [print(m['role'], '→', m['content'][:60]) for m in d]"
```

Expected:
- `empty history`: `[]`
- `ask`: ответ есть, iterations>=1, события из спека.
- `history after`: 2 сообщения (user + assistant).

- [ ] **Step 5: Commit**

```bash
git add web/server.py
git commit -m "$(cat <<'EOF'
feat(agent): POST /api/agent/ask + GET /api/agent/messages

Non-streaming endpoint прогоняет run_collect и сохраняет user+assistant
пару в agent_messages (с trace в JSONB). /messages отдаёт глобальный поток
для UI hydration после reload'а.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Streaming endpoint `GET /api/agent/ask/stream`

**Files:**
- Modify: `web/server.py`

- [ ] **Step 1: Добавить SSE-эндпоинт**

В той же agent-секции `web/server.py` (после `agent_messages`) добавить:

```python
@app.get("/api/agent/ask/stream")
async def agent_ask_stream(query: str) -> StreamingResponse:
    """SSE-стрим events во время прогона графа. По завершении сохраняет
    user+assistant пару в БД (даже если клиент успел отсоединиться —
    через try/finally в event_source, по тому же паттерну что в RAG-стриме).
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="query пуст")

    _save_agent_message("user", query, None)

    async def event_source():
        full_trace: list[dict] = []
        final_answer = ""
        try:
            async for event in run_stream(query):
                full_trace.append(event)
                if event["type"] == "final_answer":
                    final_answer = event["data"]["text"]
                yield _sse_event(event["type"], event["data"])
        finally:
            # Сохраняем даже при дисконнекте клиента — иначе ответ
            # «уехал» в UI, но в БД его нет.
            if full_trace:
                _save_agent_message("assistant", final_answer, full_trace)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

`_sse_event` уже определена в `web/server.py` (выше, в RAG-секции) — переиспользуем её.

- [ ] **Step 2: Перезапустить и smoke**

```bash
pkill -f "uvicorn web.server" 2>/dev/null; sleep 1
.venv/bin/uvicorn web.server:app --host 0.0.0.0 --port 8000
```
(background)

```bash
until curl -s http://localhost:8000/api/health 2>/dev/null | grep -q '"ok":true'; do sleep 0.5; done

curl -s -N "http://localhost:8000/api/agent/ask/stream?query=$(python3 -c 'import urllib.parse; print(urllib.parse.quote("Какая сегодня погода?"))')" --max-time 120 2>&1 | head -30
```

Expected: видишь поток `event: node_start\ndata: {...}\n\n`, `event: tool_call\n...`, и т.д. — заканчивающийся `event: done` или `event: error`.

- [ ] **Step 3: Проверить, что save сработал**

```bash
curl -s http://localhost:8000/api/agent/messages | python3 -c "import json,sys; print('total:', len(json.load(sys.stdin)))"
```

Expected: число > предыдущего (за счёт user+assistant пары из стрима).

- [ ] **Step 4: Commit**

```bash
git add web/server.py
git commit -m "$(cat <<'EOF'
feat(agent): GET /api/agent/ask/stream (SSE)

Стримим events по мере прогона графа. Save assistant-сообщения в
try/finally — переживает дисконнект клиента (тот же паттерн что
ранее починили в RAG-стриме).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Frontend — types, api client, store

**Files:**
- Create: `frontend/src/lib/agent-types.ts`
- Create: `frontend/src/lib/agent-api.ts`
- Create: `frontend/src/stores/agent-ui.ts`

- [ ] **Step 1: Создать `frontend/src/lib/agent-types.ts`**

```typescript
export type AgentEventType =
  | "node_start"
  | "tool_call"
  | "tool_result"
  | "final_answer"
  | "done"
  | "error";

export interface TraceEvent {
  type: AgentEventType;
  timestamp: string;
  data: Record<string, unknown>;
}

export interface AgentAskResponse {
  answer: string;
  trace: TraceEvent[];
  iterations: number;
}

export interface AgentMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  trace: TraceEvent[] | null;
  created_at: string;
}
```

- [ ] **Step 2: Создать `frontend/src/lib/agent-api.ts`**

```typescript
import type { AgentAskResponse, AgentMessage } from "./agent-types";

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/$/, "");

function apiUrl(path: string): string {
  return API_BASE + path;
}

async function json<T>(input: string, init?: RequestInit): Promise<T> {
  const res = await fetch(input, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    let detail: string | undefined;
    try {
      const body = await res.json();
      detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body?.detail);
    } catch {}
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const agentApi = {
  ask: (query: string) =>
    json<AgentAskResponse>(apiUrl("/api/agent/ask"), {
      method: "POST",
      body: JSON.stringify({ query }),
    }),

  messages: (limit = 200) =>
    json<AgentMessage[]>(apiUrl(`/api/agent/messages?limit=${limit}`)),

  /** EventSource не поддерживает POST — query улетает в query-string. */
  buildStreamUrl: (query: string) => {
    const params = new URLSearchParams({ query });
    return apiUrl(`/api/agent/ask/stream?${params.toString()}`);
  },
};
```

- [ ] **Step 3: Создать `frontend/src/stores/agent-ui.ts`**

```typescript
import { create } from "zustand";
import type { TraceEvent } from "@/lib/agent-types";

/** Draft assistant-сообщения во время стрима (накапливающийся trace). */
export interface AgentDraft {
  userQuery: string;
  trace: TraceEvent[];
  answer: string;       // обновляется на final_answer event
  finished: boolean;
  abort: (() => void) | null;
}

interface AgentUIState {
  draft: AgentDraft | null;
  setDraft: (d: AgentDraft | null) => void;
  appendTrace: (ev: TraceEvent) => void;
  setAnswer: (text: string) => void;
  setFinished: (v: boolean) => void;
}

export const useAgentUi = create<AgentUIState>((set) => ({
  draft: null,
  setDraft: (d) => set({ draft: d }),
  appendTrace: (ev) =>
    set((s) => {
      if (!s.draft) return s;
      return { draft: { ...s.draft, trace: [...s.draft.trace, ev] } };
    }),
  setAnswer: (text) =>
    set((s) => (s.draft ? { draft: { ...s.draft, answer: text } } : s)),
  setFinished: (v) =>
    set((s) => (s.draft ? { draft: { ...s.draft, finished: v } } : s)),
}));
```

- [ ] **Step 4: Type-check**

```bash
cd frontend && npx tsc --noEmit
```

Expected: без ошибок.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/agent-types.ts frontend/src/lib/agent-api.ts frontend/src/stores/agent-ui.ts
git commit -m "$(cat <<'EOF'
feat(frontend): agent types, api client, zustand store

Зеркало Pydantic-моделей бэкенда + Zustand-store для draft'а во время
стрима (накапливающийся trace).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Sidebar tabs (Чаты / Agent)

**Files:**
- Create (если нет): `frontend/src/components/ui/tabs.tsx`
- Modify: `frontend/src/components/sidebar/sidebar.tsx`

- [ ] **Step 1: Проверить наличие `tabs.tsx`**

```bash
ls frontend/src/components/ui/tabs.tsx 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

Если MISSING — создать `frontend/src/components/ui/tabs.tsx`:

```tsx
"use client";

import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import { cn } from "@/lib/utils";

const Tabs = TabsPrimitive.Root;

const TabsList = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    ref={ref}
    className={cn(
      "inline-flex h-9 items-center justify-center rounded-md bg-muted p-0.5 text-muted-foreground",
      className,
    )}
    {...props}
  />
));
TabsList.displayName = TabsPrimitive.List.displayName;

const TabsTrigger = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Trigger
    ref={ref}
    className={cn(
      "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-2 py-1 text-xs font-medium ring-offset-background transition-all data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow",
      className,
    )}
    {...props}
  />
));
TabsTrigger.displayName = TabsPrimitive.Trigger.displayName;

const TabsContent = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Content ref={ref} className={cn("mt-2", className)} {...props} />
));
TabsContent.displayName = TabsPrimitive.Content.displayName;

export { Tabs, TabsList, TabsTrigger, TabsContent };
```

И установить пакет если отсутствует:

```bash
cd frontend && grep -q "@radix-ui/react-tabs" package.json || npm install @radix-ui/react-tabs
```

- [ ] **Step 2: Обновить `frontend/src/components/sidebar/sidebar.tsx`**

Текущий файл сейчас имеет ChatsList + SourcesPanel в вертикальном split'е без табов. Заменить middle-секцию на табы.

Полный новый файл:

```tsx
"use client";

import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { Activity, Sparkles } from "lucide-react";
import { ChatsList } from "./chats-list";
import { SourcesPanel } from "./sources-panel";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";

export function Sidebar() {
  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
  });

  return (
    <aside className="flex flex-col h-full border-r border-border bg-card/40 w-72 min-w-[18rem]">
      <header className="px-4 py-3 border-b border-border">
        <h1 className="text-sm font-semibold tracking-tight">RAG Studio</h1>
        <p className="text-[11px] text-muted-foreground">LM Studio + pgvector</p>
      </header>

      <Tabs defaultValue="chats" className="flex-1 flex flex-col min-h-0">
        <TabsList className="grid grid-cols-2 mx-3 mt-2">
          <TabsTrigger value="chats">Чаты</TabsTrigger>
          <TabsTrigger value="agent">Agent</TabsTrigger>
        </TabsList>

        <TabsContent value="chats" className="flex-1 min-h-0 flex flex-col mt-0">
          <div className="flex-[2] min-h-0">
            <ChatsList />
          </div>
          <div className="flex-[3] min-h-0">
            <SourcesPanel />
          </div>
        </TabsContent>

        <TabsContent value="agent" className="flex-1 min-h-0 mt-0">
          <div className="flex flex-col items-center justify-center h-full p-4 gap-3 text-center">
            <Sparkles className="h-6 w-6 text-primary" />
            <p className="text-xs text-muted-foreground">
              Спорт-консьерж Рондо.
              <br />
              Спросит погоду и проверит свободные корты.
            </p>
            <Link href="/agent">
              <Button size="sm" variant="outline" className="text-xs">
                Открыть Agent
              </Button>
            </Link>
          </div>
        </TabsContent>
      </Tabs>

      <footer className="px-3 py-2 border-t border-border flex items-center justify-between text-[11px] text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <Activity className="h-3 w-3" />
          {healthQuery.data ? `${healthQuery.data.chunks_in_db} чанков` : "…"}
        </span>
        {healthQuery.isError && <span className="text-amber-400">backend offline</span>}
      </footer>
    </aside>
  );
}
```

- [ ] **Step 3: Type-check**

```bash
cd frontend && npx tsc --noEmit
```

Expected: без ошибок.

- [ ] **Step 4: Smoke в браузере**

Открыть http://localhost:3000 → видишь две вкладки «Чаты» / «Agent». На «Agent» — CTA с кнопкой «Открыть Agent» ведущей на `/agent` (страницу ещё нет, 404 нормально пока).

- [ ] **Step 5: Commit**

Если `tabs.tsx` создан + `package.json` менялся — в add'е они тоже:

```bash
git add frontend/src/components/sidebar/sidebar.tsx \
        frontend/src/components/ui/tabs.tsx \
        frontend/package.json frontend/package-lock.json
git commit -m "$(cat <<'EOF'
feat(frontend): sidebar tabs (Чаты / Agent) + tabs ui primitive

Tabs из shadcn (radix-tabs обёртка). На таб «Agent» — CTA-плашка
с переходом на /agent.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Если `tabs.tsx` уже был — не включай его и package-файлы в add.

---

### Task 12: trace-step + trace-timeline компоненты

**Files:**
- Create: `frontend/src/components/agent/trace-step.tsx`
- Create: `frontend/src/components/agent/trace-timeline.tsx`

- [ ] **Step 1: Создать `trace-step.tsx`**

```tsx
"use client";

import * as React from "react";
import { ChevronRight, Wrench, MessageSquare, Brain, CheckCircle2, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { TraceEvent } from "@/lib/agent-types";

interface Props {
  event: TraceEvent;
}

function eventIcon(type: TraceEvent["type"]) {
  switch (type) {
    case "node_start":   return Brain;
    case "tool_call":    return Wrench;
    case "tool_result":  return CheckCircle2;
    case "final_answer": return MessageSquare;
    case "done":         return CheckCircle2;
    case "error":        return AlertCircle;
  }
}

function eventTitle(ev: TraceEvent): string {
  switch (ev.type) {
    case "node_start": {
      const node = ev.data.node as string;
      return node === "agent" ? "Думаю..." : `Узел: ${node}`;
    }
    case "tool_call": {
      const name = ev.data.name as string;
      const args = ev.data.args as Record<string, unknown>;
      const argStr = Object.entries(args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
      return `🔧 ${name}(${argStr})`;
    }
    case "tool_result": {
      const name = ev.data.name as string;
      const r = ev.data.result as Record<string, unknown> | string;
      if (typeof r === "object" && r !== null) {
        if ("error" in r) return `${name} → ошибка: ${r.error}`;
        if ("summary" in r) return `${name} → ${r.summary}`;
      }
      return `${name} → готово`;
    }
    case "final_answer": return "Финальный ответ";
    case "done":         return `Готово (итераций: ${ev.data.iterations})`;
    case "error":        return `Ошибка: ${ev.data.message}`;
  }
}

function eventColor(type: TraceEvent["type"]): string {
  switch (type) {
    case "error":         return "text-red-400";
    case "final_answer":  return "text-primary";
    case "tool_call":     return "text-amber-400";
    case "tool_result":   return "text-emerald-400";
    default:              return "text-muted-foreground";
  }
}

export function TraceStep({ event }: Props) {
  const [open, setOpen] = React.useState(false);
  const Icon = eventIcon(event.type);
  const title = eventTitle(event);
  const color = eventColor(event.type);

  return (
    <div className="flex flex-col text-[11px]">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={cn(
          "flex items-start gap-1.5 text-left hover:bg-accent/40 rounded px-1 py-0.5 transition-colors",
          color,
        )}
      >
        <ChevronRight className={cn("h-3 w-3 mt-0.5 shrink-0 transition-transform", open && "rotate-90")} />
        <Icon className="h-3 w-3 mt-0.5 shrink-0" />
        <span className="break-words">{title}</span>
      </button>
      {open && (
        <pre className="ml-5 mt-1 p-2 rounded bg-card/60 border border-border overflow-x-auto text-[10px] leading-tight">
          {JSON.stringify(event.data, null, 2)}
        </pre>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Создать `trace-timeline.tsx`**

```tsx
"use client";

import * as React from "react";
import type { TraceEvent } from "@/lib/agent-types";
import { TraceStep } from "./trace-step";

interface Props {
  trace: TraceEvent[];
}

export function TraceTimeline({ trace }: Props) {
  if (!trace.length) return null;
  return (
    <div className="flex flex-col gap-0.5 py-2">
      {trace.map((ev, i) => (
        <TraceStep key={i} event={ev} />
      ))}
    </div>
  );
}
```

- [ ] **Step 3: Type-check**

```bash
cd frontend && npx tsc --noEmit
```

Expected: без ошибок.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/agent/trace-step.tsx frontend/src/components/agent/trace-timeline.tsx
git commit -m "$(cat <<'EOF'
feat(frontend): agent TraceStep + TraceTimeline components

TraceStep — одна collapsible-строка с иконкой/цветом по type и
разворачивающимся JSON-pre. TraceTimeline — список без обвязки.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: AgentChat + /agent route page

**Files:**
- Create: `frontend/src/components/agent/agent-chat.tsx`
- Create: `frontend/src/app/agent/page.tsx`

- [ ] **Step 1: Создать `agent-chat.tsx`**

```tsx
"use client";

import * as React from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Send, StopCircle, Bot, User, Sparkles } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { agentApi } from "@/lib/agent-api";
import type { AgentMessage, TraceEvent } from "@/lib/agent-types";
import { useAgentUi } from "@/stores/agent-ui";
import { TraceTimeline } from "./trace-timeline";

export function AgentChat() {
  const queryClient = useQueryClient();
  const draft = useAgentUi((s) => s.draft);
  const setDraft = useAgentUi((s) => s.setDraft);
  const appendTrace = useAgentUi((s) => s.appendTrace);
  const setAnswer = useAgentUi((s) => s.setAnswer);
  const setFinished = useAgentUi((s) => s.setFinished);

  const [text, setText] = React.useState("");
  const bottomRef = React.useRef<HTMLDivElement>(null);

  const messagesQuery = useQuery({
    queryKey: ["agent", "messages"],
    queryFn: () => agentApi.messages(),
  });

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messagesQuery.data?.length, draft?.trace.length]);

  const send = React.useCallback(() => {
    const q = text.trim();
    if (!q || draft) return;
    setText("");
    setDraft({ userQuery: q, trace: [], answer: "", finished: false, abort: null });

    const url = agentApi.buildStreamUrl(q);
    const es = new EventSource(url);
    const abort = () => es.close();
    setDraft({ userQuery: q, trace: [], answer: "", finished: false, abort });

    const onEvent = (type: TraceEvent["type"]) => (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data);
        const ev: TraceEvent = { type, timestamp: new Date().toISOString(), data };
        appendTrace(ev);
        if (type === "final_answer" && typeof data.text === "string") {
          setAnswer(data.text);
        }
        if (type === "done" || type === "error") {
          setFinished(true);
          es.close();
          queryClient.invalidateQueries({ queryKey: ["agent", "messages"] });
          // Через 800ms убираем draft — к этому моменту /messages уже отдаст
          // assistant-сообщение, отсорсенное из БД.
          setTimeout(() => setDraft(null), 800);
        }
      } catch (err) {
        console.error("SSE parse failed", err);
      }
    };

    es.addEventListener("node_start", onEvent("node_start"));
    es.addEventListener("tool_call", onEvent("tool_call"));
    es.addEventListener("tool_result", onEvent("tool_result"));
    es.addEventListener("final_answer", onEvent("final_answer"));
    es.addEventListener("done", onEvent("done"));
    es.addEventListener("error", (e) => {
      // Стандартный EventSource error event (без data) — это разрыв связи.
      const msgEv = e as MessageEvent;
      if (msgEv.data) {
        onEvent("error")(msgEv);
      } else {
        toast.error("SSE-соединение прервано");
        setFinished(true);
        es.close();
        setDraft(null);
      }
    });
  }, [text, draft, setDraft, appendTrace, setAnswer, setFinished, queryClient]);

  const stop = React.useCallback(() => {
    draft?.abort?.();
    setDraft(null);
  }, [draft, setDraft]);

  const messages: AgentMessage[] = messagesQuery.data ?? [];
  const isStreaming = draft !== null && !draft.finished;

  return (
    <div className="flex h-full flex-col">
      <div className="px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" />
          <h2 className="text-sm font-semibold">Спорт-консьерж Рондо</h2>
        </div>
        <p className="text-[11px] text-muted-foreground">
          Спрашивай про погоду и свободные корты — агент сам подёргает API.
        </p>
      </div>

      <ScrollArea className="flex-1">
        <div className="mx-auto max-w-3xl px-4 py-6 space-y-6">
          {messages.length === 0 && !draft && (
            <div className="flex flex-col items-center text-center py-16 gap-2">
              <Sparkles className="h-8 w-8 text-primary" />
              <p className="text-sm text-muted-foreground max-w-md">
                Спроси, например: <br />
                «Найди солнечный день на следующей неделе со свободным кортом».
              </p>
            </div>
          )}

          {messages.map((m) => (
            <Bubble
              key={m.id}
              role={m.role}
              content={m.content}
              trace={m.trace}
            />
          ))}

          {draft && (
            <>
              <Bubble role="user" content={draft.userQuery} />
              <Bubble
                role="assistant"
                content={draft.answer || (draft.finished ? "(пусто)" : "Думаю…")}
                trace={draft.trace}
                streaming={isStreaming}
              />
            </>
          )}

          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      <div className="border-t border-border bg-background p-4">
        <div className="mx-auto max-w-3xl flex items-end gap-2 rounded-2xl border border-border bg-card p-2">
          <Textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Спроси что-нибудь… (Enter — отправить)"
            className="flex-1 min-h-[40px] max-h-[200px] resize-none border-0 bg-transparent shadow-none focus-visible:ring-0 px-2 py-2"
            rows={1}
          />
          {isStreaming ? (
            <Button variant="destructive" size="icon" onClick={stop} title="Прервать" className="h-9 w-9 shrink-0">
              <StopCircle className="h-4 w-4" />
            </Button>
          ) : (
            <Button size="icon" onClick={send} disabled={!text.trim()} title="Отправить" className="h-9 w-9 shrink-0">
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

function Bubble({
  role,
  content,
  trace,
  streaming,
}: {
  role: "user" | "assistant";
  content: string;
  trace?: TraceEvent[] | null;
  streaming?: boolean;
}) {
  return (
    <div className={cn("flex gap-3", role === "user" ? "flex-row-reverse" : "flex-row")}>
      <div className={cn(
        "h-7 w-7 shrink-0 rounded-full flex items-center justify-center border",
        role === "user"
          ? "bg-primary/10 border-primary/40 text-primary"
          : "bg-secondary border-border text-foreground",
      )}>
        {role === "user" ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}
      </div>
      <div className="max-w-[calc(100%-3rem)] flex-1 min-w-0">
        <div className={cn(
          "rounded-2xl px-4 py-3",
          role === "user" ? "bg-primary/10 border border-primary/30" : "bg-card border border-border",
        )}>
          <p className="prose-rag whitespace-pre-wrap">{content}</p>
          {streaming && <span className="ml-1 inline-block w-[6px] h-[1em] bg-primary animate-pulse align-middle" />}
          {role === "assistant" && trace && trace.length > 0 && (
            <div className="mt-2 border-t border-border pt-2">
              <TraceTimeline trace={trace} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Создать `frontend/src/app/agent/page.tsx`**

```tsx
import { AgentChat } from "@/components/agent/agent-chat";

export default function AgentPage() {
  return <AgentChat />;
}
```

`mkdir -p frontend/src/app/agent` сделай если папки нет.

- [ ] **Step 3: Type-check**

```bash
cd frontend && npx tsc --noEmit
```

Expected: без ошибок.

- [ ] **Step 4: Smoke в браузере**

Открыть http://localhost:3000/agent → видишь chat-интерфейс с placeholder'ом. Композер внизу. Hard-reload (Cmd+Shift+R) — `messages` должны загрузиться (если запускал curl-ask раньше, увидишь старые сообщения).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/agent/agent-chat.tsx frontend/src/app/agent/page.tsx
git commit -m "$(cat <<'EOF'
feat(frontend): /agent route с AgentChat + EventSource-стримом

Bubble assistant'а имеет встроенный TraceTimeline (после ответа или
во время стрима). EventSource слушает 6 типов событий, копит в Zustand,
по done/error invalidate-queries и убирает draft через 800ms (даёт
refetch'у вернуть assistant из БД).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: End-to-end manual verification

Не пишет код — финальная проверка.

- [ ] **Step 1: Простой weather-only**

Открыть http://localhost:3000/agent. Спросить:

> Какая погода 1 июня?

Ожидание: trace показывает `node_start: agent` → `tool_call: get_weather(date="2026-06-01")` → `node_start: tools` → `tool_result: get_weather → ...` → `node_start: agent` → `final_answer` → `done`. Ответ упоминает температуру/условия.

- [ ] **Step 2: Courts-only**

> Что свободно 1 июня?

Ожидание: один `tool_call: get_courts_availability`, ответ со слотами.

- [ ] **Step 3: Главный кейс — связка**

> Найди солнечный день на следующей неделе со свободным кортом

Ожидание: ~14 tool_calls (7 дат × 2 tool'а), либо итеративные (по дате за раз). Ответ с конкретными датами/окнами. Может занять 30-120 секунд. **Если LLM зациклится — увидишь `error: max_iter` после 10 итераций. Это валидный результат теста: см. Spec «Известные риски».**

- [ ] **Step 4: Out-of-scope**

> Что в кино идёт?

Ожидание: LLM не зовёт tools, отвечает что не его зона.

- [ ] **Step 5: Tool-error**

Временно в `.env` поменять:
```
SKKRONDO_BASE_URL=http://localhost:1
```

Перезапустить uvicorn. Спросить «погоду» — в trace должен быть `tool_result` с `error: ...`, LLM в финальном ответе должен упомянуть что данные недоступны.

Вернуть обратно:
```
SKKRONDO_BASE_URL=https://api.skkrondo.ru
```

- [ ] **Step 6: Reload-persistence**

После любого успешного ask — F5 → видишь те же messages (читаются из `agent_messages` через `GET /api/agent/messages`).

- [ ] **Step 7: RAG-регресс**

Перейти на таб «Чаты», создать чат, спросить «Что такое HNSW?» — RAG-цепочка отрабатывает как раньше. Это проверка что добавление агента не сломало основной флоу.

- [ ] **Step 8: Финальный git-лог**

```bash
git log --oneline 6463bc3..HEAD
```

Ожидание: 13 коммитов (T1-T13) без regression'ов.

---

## Self-review

**Spec coverage:**

- ✅ Tools `get_weather` + `get_courts_availability` — Task 4
- ✅ LangGraph граф (agent + ToolNode + tools_condition) — Task 6
- ✅ MAX_ITER safety net + system prompt с today — Tasks 3, 6, 7
- ✅ Runner с 6 event types — Task 7
- ✅ Backend POST /ask, GET /ask/stream, GET /messages — Tasks 8, 9
- ✅ Storage `agent_messages` table + save logic — Tasks 2, 8
- ✅ Frontend types/api/store — Task 10
- ✅ Sidebar tabs — Task 11
- ✅ TraceTimeline/TraceStep — Task 12
- ✅ /agent page + AgentChat с EventSource — Task 13
- ✅ Manual scenarios (6 из spec'а + RAG-регресс) — Task 14

**Placeholder scan:** в плане нет TBD/«сделать позже». Все code-блоки выписаны целиком.

**Type consistency:**
- `AgentState` определён в Task 3, используется в Task 6
- `TOOLS` экспортирован из Task 4, импортирован в Task 6
- `build_graph` экспортирован из Task 6, импортирован в Task 7
- `run_stream` / `run_collect` экспортированы из Task 7, импортированы в Task 8
- `_save_agent_message` определён в Task 8, переиспользован в Task 9
- `AgentMessage` / `TraceEvent` типы в Task 10, использованы в Tasks 12, 13
- `agentApi` определён в Task 10, использован в Task 13
- `useAgentUi` определён в Task 10, использован в Task 13
- `TraceTimeline` определён в Task 12, использован в Task 13

Согласованность OK.

**Risks:**
- Если Gemma 4 e2b плохо tool-callит — это будет видно в T14 сценариях. Mitigation в spec'е.
- Если `langgraph` / `langchain-openai` потянут конфликт зависимостей — придётся отдельно резолвить версии.
- На стрим-режиме SSE через Next.js proxy может буферизоваться. Используется `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` напрямую (как у RAG-стрима) — этот env уже выставлен в проекте.
