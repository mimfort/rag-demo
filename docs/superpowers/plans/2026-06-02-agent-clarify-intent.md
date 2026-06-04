# Agent Clarify Intent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Первым шагом каждого turn'а агент перефразирует запрос пользователя (резолвит относительные даты в ISO), показывает «Вы имели в виду: …? Да/Нет» и идёт работать только после подтверждения; на «нет» — цикл переформулировок.

**Architecture:** В LangGraph-граф перед `agent ⇄ tools` добавляются два узла — `interpret` (один LLM-вызов, перефразировка) и `confirm` (`interrupt()` + пауза). Граф компилируется с `MemorySaver`; пауза/возобновление идут через `thread_id` и `Command(resume=...)`. Бэкенд отдаёт новое SSE-событие `clarify`; фронт рисует карточку подтверждения и шлёт resume.

**Tech Stack:** Python 3.13, LangGraph 0.6.11, langchain-core, FastAPI (SSE), pytest (новое), Next.js + React + zustand + EventSource.

**Спека:** `docs/superpowers/specs/2026-06-02-agent-clarify-intent-design.md`

---

## Структура файлов

**Бэкенд:**
- `agent/clarify.py` — **новый**: промпт интерпретатора + `interpret_query()` (тестируемая чистая обёртка над LLM).
- `agent/state.py` — новые поля state.
- `agent/config.py` — `MAX_CLARIFY_ROUNDS`.
- `agent/graph.py` — узлы `interpret`/`confirm`, conditional edges, `agent_node` использует `effective_query`, компиляция с `MemorySaver`, `build_graph(plain_llm, agent_llm, checkpointer)` с инъекцией для тестов.
- `agent/runner.py` — `_astream_events(thread_id, *, query/resume, graph)`, обработка `__interrupt__` → `clarify`, `run_collect` авто-подтверждает.
- `web/server.py` — генерация `thread_id`, эндпоинт `/api/agent/resume/stream`, событие `clarify`, in-memory накопление trace по `thread_id`, сохранение на `done`.

**Тесты:**
- `tests/conftest.py` — **новый**: `FakeChat` stub.
- `tests/test_clarify.py` — **новый**: `interpret_query`, узлы.
- `tests/test_graph_interrupt.py` — **новый**: interrupt/resume/цикл/cap.
- `tests/test_runner_clarify.py` — **новый**: `__interrupt__` → событие `clarify`.

**Фронтенд:**
- `frontend/src/lib/agent-types.ts` — событие `clarify`.
- `frontend/src/lib/agent-api.ts` — `buildResumeUrl`.
- `frontend/src/stores/agent-ui.ts` — `pendingClarify`.
- `frontend/src/components/agent/clarify-prompt.tsx` — **новый** компонент.
- `frontend/src/components/agent/agent-chat.tsx` — `openStream`, рендер карточки.
- `frontend/src/components/agent/trace-step.tsx` — отрисовка шага `clarify`.

---

## Task 0: Тестовая инфраструктура

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pytest.ini`

- [ ] **Step 1: Добавить pytest в requirements**

Дописать в конец `requirements.txt`:

```
pytest==8.3.4
pytest-asyncio==0.25.2
```

- [ ] **Step 2: Установить**

Run: `.venv/bin/pip install pytest==8.3.4 pytest-asyncio==0.25.2`
Expected: `Successfully installed pytest-... pytest-asyncio-...`

- [ ] **Step 3: Создать `pytest.ini`**

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
```

- [ ] **Step 4: Создать `tests/__init__.py`** (пустой файл)

```python
```

- [ ] **Step 5: Создать `tests/conftest.py` с FakeChat**

```python
"""Тестовые stub'ы. FakeChat подменяет ChatOpenAI: возвращает заранее
заданные AIMessage по очереди (последний повторяется, если вызовов больше).
Покрывает и .ainvoke (узлы), и .bind_tools (agent_node)."""
from __future__ import annotations

from langchain_core.messages import AIMessage


class FakeChat:
    def __init__(self, responses):
        if not isinstance(responses, list):
            responses = [responses]
        self._responses = responses
        self._i = 0

    def _next(self) -> AIMessage:
        r = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return r if isinstance(r, AIMessage) else AIMessage(r)

    async def ainvoke(self, messages):
        return self._next()

    def invoke(self, messages):
        return self._next()

    def bind_tools(self, tools):
        return self
```

- [ ] **Step 6: Smoke-тест инфраструктуры**

Создать временно `tests/test_smoke.py`:

```python
from tests.conftest import FakeChat
from langchain_core.messages import AIMessage


async def test_fakechat_sequence():
    f = FakeChat(["a", AIMessage("b")])
    assert (await f.ainvoke([])).content == "a"
    assert (await f.ainvoke([])).content == "b"
    assert (await f.ainvoke([])).content == "b"  # последний повторяется
```

Run: `.venv/bin/pytest tests/test_smoke.py -v`
Expected: PASS (1 passed)

- [ ] **Step 7: Удалить smoke-тест и закоммитить**

```bash
rm tests/test_smoke.py
git add requirements.txt pytest.ini tests/__init__.py tests/conftest.py
git commit -m "test(agent): pytest-инфраструктура и FakeChat stub"
```

---

## Task 1: Расширить AgentState

**Files:**
- Modify: `agent/state.py`

- [ ] **Step 1: Добавить поля в AgentState**

Заменить класс `AgentState` в `agent/state.py` на:

```python
class AgentState(TypedDict, total=False):
    # `add_messages` — встроенный reducer LangGraph: новые сообщения
    # АППЕНДЯТСЯ к существующим. Каноничный ReAct-state.
    messages: Annotated[list[BaseMessage], add_messages]
    # Последняя предложенная перефразировка запроса (узел interpret).
    interpretation: str
    # Подтверждённая формулировка — её использует agent_node.
    effective_query: str
    # Последнее уточнение пользователя на «нет» (для нового круга).
    correction: str | None
    # Счётчик кругов clarify — гард против бесконечного цикла.
    clarify_rounds: int
```

`total=False` — поля кроме `messages` необязательны во входном state (граф стартует с одним `messages`); узлы читают их через `state.get(...)`.

- [ ] **Step 2: Проверка импорта**

Run: `.venv/bin/python -c "from agent.state import AgentState; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add agent/state.py
git commit -m "feat(agent): поля clarify в AgentState"
```

---

## Task 2: Узел interpret — промпт и interpret_query

**Files:**
- Create: `agent/clarify.py`
- Create: `tests/test_clarify.py`

- [ ] **Step 1: Написать падающий тест**

`tests/test_clarify.py`:

```python
from agent.clarify import build_interpreter_messages, interpret_query
from tests.conftest import FakeChat


def test_messages_include_today_and_query():
    msgs = build_interpreter_messages("какая погода сегодня", None, "2026-06-02")
    assert msgs[0].type == "system"
    assert "2026-06-02" in msgs[0].content
    assert "какая погода сегодня" in msgs[1].content


def test_messages_include_correction():
    msgs = build_interpreter_messages("погода", "я про завтра", "2026-06-02")
    assert "я про завтра" in msgs[1].content


async def test_interpret_query_returns_stripped_content():
    llm = FakeChat("  какая погода будет 2026-06-02  ")
    out = await interpret_query(llm, "какая погода сегодня", None, "2026-06-02")
    assert out == "какая погода будет 2026-06-02"
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `.venv/bin/pytest tests/test_clarify.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'agent.clarify')

- [ ] **Step 3: Реализовать `agent/clarify.py`**

```python
"""
Шаг интерпретации запроса (clarify). Один LLM-вызов: перефразирует запрос
пользователя в одну чёткую фразу, резолвит относительные даты в ISO.

Вынесено из graph.py отдельным модулем — чтобы логику можно было покрыть
unit-тестами без сборки всего графа.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage


INTERPRETER_PROMPT = """Ты — помощник по СКК «Рондо». Твоя задача — НЕ отвечать на вопрос, \
а переформулировать запрос пользователя в одну короткую чёткую фразу, по которой потом \
будет работать агент (он умеет получать погоду и занятость кортов на конкретные даты).

Сегодня: {today}.

Правила:
- Резолви относительные даты в конкретные ISO-даты (YYYY-MM-DD): «сегодня» → {today}, \
а также «завтра», «на выходных», «следующая неделя» → конкретные даты/диапазон.
- Сохрани суть запроса (погода / занятость кортов / и то и другое).
- Если пользователь прислал уточнение к прошлой попытке — обязательно учти его.
- Верни ТОЛЬКО переформулированную фразу. Без пояснений, без кавычек, без префиксов."""


def build_interpreter_messages(
    original: str, correction: str | None, today: str
) -> list:
    """Собирает messages для LLM-интерпретатора."""
    system = SystemMessage(INTERPRETER_PROMPT.format(today=today))
    if correction:
        human = HumanMessage(
            f"Исходный запрос пользователя: {original}\n"
            f"Моя прошлая формулировка оказалась неверной. "
            f"Пользователь уточнил: {correction}\n"
            f"Переформулируй заново с учётом уточнения."
        )
    else:
        human = HumanMessage(f"Запрос пользователя: {original}")
    return [system, human]


async def interpret_query(
    llm, original: str, correction: str | None, today: str
) -> str:
    """Зовёт LLM и возвращает перефразировку (без обрамляющих пробелов)."""
    messages = build_interpreter_messages(original, correction, today)
    response = await llm.ainvoke(messages)
    return (response.content or "").strip()
```

- [ ] **Step 4: Запустить — убедиться, что проходит**

Run: `.venv/bin/pytest tests/test_clarify.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/clarify.py tests/test_clarify.py
git commit -m "feat(agent): interpret_query — перефразировка запроса"
```

---

## Task 3: MAX_CLARIFY_ROUNDS в config

**Files:**
- Modify: `agent/config.py`

- [ ] **Step 1: Добавить константу**

В `agent/config.py` после строки с `MAX_ITER` добавить:

```python
# Гард против бесконечного цикла подтверждения (interpret ⇄ confirm).
# После стольких кругов «нет» агент принудительно принимает последнюю
# интерпретацию и идёт работать.
MAX_CLARIFY_ROUNDS: int = 5
```

- [ ] **Step 2: Проверка**

Run: `.venv/bin/python -c "from agent.config import MAX_CLARIFY_ROUNDS; print(MAX_CLARIFY_ROUNDS)"`
Expected: `5`

- [ ] **Step 3: Commit**

```bash
git add agent/config.py
git commit -m "feat(agent): MAX_CLARIFY_ROUNDS"
```

---

## Task 4: Граф — узлы interpret/confirm, маршрутизация, checkpointer

**Files:**
- Modify: `agent/graph.py`
- Create: `tests/test_graph_interrupt.py`

- [ ] **Step 1: Написать падающий тест interrupt/resume**

`tests/test_graph_interrupt.py`:

```python
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent.graph import build_graph
from tests.conftest import FakeChat


def _build(plain_responses, agent_responses):
    return build_graph(
        plain_llm=FakeChat(plain_responses),
        agent_llm=FakeChat(agent_responses),
        checkpointer=MemorySaver(),
    )


async def _run_to_first_interrupt(graph, query, thread_id):
    cfg = {"configurable": {"thread_id": thread_id}}
    interrupts = []
    async for update in graph.astream(
        {"messages": [HumanMessage(query)]}, cfg, stream_mode="updates"
    ):
        if "__interrupt__" in update:
            interrupts.append(update["__interrupt__"][0].value)
    return interrupts


async def test_start_interrupts_with_interpretation():
    graph = _build("какая погода будет 2026-06-02", AIMessage("Готово"))
    interrupts = await _run_to_first_interrupt(graph, "погода сегодня", "t1")
    assert len(interrupts) == 1
    assert interrupts[0]["interpretation"] == "какая погода будет 2026-06-02"
    assert interrupts[0]["original"] == "погода сегодня"
    assert interrupts[0]["round"] == 1


async def test_resume_confirmed_reaches_agent():
    graph = _build("какая погода будет 2026-06-02", AIMessage("Солнечно, +17"))
    cfg = {"configurable": {"thread_id": "t2"}}
    await _run_to_first_interrupt(graph, "погода сегодня", "t2")
    # Подтверждаем — граф должен дойти до финального ответа агента.
    final = None
    async for update in graph.astream(
        Command(resume={"confirmed": True, "correction": None}),
        cfg, stream_mode="updates",
    ):
        if "agent" in update:
            final = update["agent"]["messages"][-1].content
    assert final == "Солнечно, +17"


async def test_resume_rejected_reinterprets():
    graph = _build(
        ["погода 2026-06-02", "погода 2026-06-03"], AIMessage("Готово")
    )
    cfg = {"configurable": {"thread_id": "t3"}}
    await _run_to_first_interrupt(graph, "погода", "t3")
    second = []
    async for update in graph.astream(
        Command(resume={"confirmed": False, "correction": "я про завтра"}),
        cfg, stream_mode="updates",
    ):
        if "__interrupt__" in update:
            second.append(update["__interrupt__"][0].value)
    assert len(second) == 1
    assert second[0]["interpretation"] == "погода 2026-06-03"
    assert second[0]["round"] == 2


async def test_max_rounds_auto_accepts():
    from agent.config import MAX_CLARIFY_ROUNDS
    graph = _build("погода 2026-06-02", AIMessage("Готово"))
    cfg = {"configurable": {"thread_id": "t4"}}
    await _run_to_first_interrupt(graph, "погода", "t4")
    reached_agent = False
    # MAX_CLARIFY_ROUNDS-1 раз отвечаем «нет», на последнем круге авто-приём.
    for _ in range(MAX_CLARIFY_ROUNDS):
        async for update in graph.astream(
            Command(resume={"confirmed": False, "correction": "не то"}),
            cfg, stream_mode="updates",
        ):
            if "agent" in update:
                reached_agent = True
    assert reached_agent
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `.venv/bin/pytest tests/test_graph_interrupt.py -v`
Expected: FAIL (TypeError: build_graph() got an unexpected keyword argument 'plain_llm')

- [ ] **Step 3: Переписать `agent/graph.py`**

Полностью заменить содержимое `agent/graph.py` на:

```python
"""
Сборка LangGraph: clarify-фаза + ReAct-цикл.

Структура:
    START → interpret → confirm ──(да)──→ agent ⇄ tools → END
                ↑                  │
                └──────(нет)───────┘

- interpret: один LLM-вызов, перефразирует запрос (резолвит даты).
- confirm:   interrupt() — пауза до подтверждения пользователя.
- agent/tools: прежний ReAct-цикл, работает по подтверждённой формулировке.
"""

from __future__ import annotations

from datetime import date

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

from agent.clarify import interpret_query
from agent.config import MAX_CLARIFY_ROUNDS
from agent.llm import make_llm
from agent.state import AgentState
from agent.tools import TOOLS


SYSTEM_PROMPT_TEMPLATE = """Ты — помощник по СКК «Рондо». Ты можешь вызывать tools чтобы получить погоду и список забронированных кортов на конкретные даты.

Сегодня: {today}.

Если пользователь спрашивает про «свободный день», «следующую неделю», «погоду» — определи диапазон дат и проверь их tool'ами. Когда соберёшь достаточно данных — отвечай по-русски, упоминай конкретные даты и часы.

Не вызывай один и тот же tool с одинаковыми аргументами повторно. Когда ответ можно дать — отвечай без новых tool_call'ов."""


def _build_system_message() -> SystemMessage:
    return SystemMessage(SYSTEM_PROMPT_TEMPLATE.format(today=date.today().isoformat()))


def _original_query(state: AgentState) -> str:
    """Контент первого human-сообщения — исходный запрос пользователя."""
    for m in state["messages"]:
        if m.type == "human":
            return m.content
    return ""


def build_graph(plain_llm=None, agent_llm=None, checkpointer=None):
    """Создаёт скомпилированный граф. Вызывается один раз на процесс.

    Параметры plain_llm/agent_llm/checkpointer — для инъекции в тестах;
    в проде остаются None и создаются реальные объекты.
    """
    plain_llm = plain_llm if plain_llm is not None else make_llm()
    agent_llm = agent_llm if agent_llm is not None else make_llm().bind_tools(TOOLS)
    checkpointer = checkpointer if checkpointer is not None else MemorySaver()

    async def interpret_node(state: AgentState) -> dict:
        interpretation = await interpret_query(
            plain_llm,
            _original_query(state),
            state.get("correction"),
            date.today().isoformat(),
        )
        return {
            "interpretation": interpretation,
            "clarify_rounds": state.get("clarify_rounds", 0) + 1,
        }

    def confirm_node(state: AgentState) -> dict:
        rounds = state.get("clarify_rounds", 0)
        # Авто-приём после лимита — иначе цикл «нет» бесконечен.
        if rounds >= MAX_CLARIFY_ROUNDS:
            return {"effective_query": state["interpretation"], "correction": None}
        # interrupt() ставит граф на паузу; value уходит клиенту, а на resume
        # сюда приходит payload из Command(resume=...).
        answer = interrupt({
            "interpretation": state["interpretation"],
            "original": _original_query(state),
            "round": rounds,
        })
        if answer.get("confirmed"):
            return {"effective_query": state["interpretation"], "correction": None}
        return {"correction": answer.get("correction") or ""}

    def route_after_confirm(state: AgentState) -> str:
        return "agent" if state.get("effective_query") else "interpret"

    async def agent_node(state: AgentState) -> dict:
        messages = list(state["messages"])
        if not messages or messages[0].type != "system":
            messages = [_build_system_message()] + messages
        eff = state.get("effective_query")
        if eff:
            # Подтверждённая формулировка подмешивается локально перед каждым
            # вызовом (в state не пишется — иначе раздувала бы историю).
            messages = messages + [SystemMessage(
                f"Подтверждённая формулировка запроса (работай строго по ней): {eff}"
            )]
        response = await agent_llm.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("interpret", interpret_node)
    graph.add_node("confirm", confirm_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("interpret")
    graph.add_edge("interpret", "confirm")
    graph.add_conditional_edges(
        "confirm", route_after_confirm,
        {"agent": "agent", "interpret": "interpret"},
    )
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
```

- [ ] **Step 4: Запустить тесты графа**

Run: `.venv/bin/pytest tests/test_graph_interrupt.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Прогнать все тесты**

Run: `.venv/bin/pytest -v`
Expected: PASS (все)

- [ ] **Step 6: Commit**

```bash
git add agent/graph.py tests/test_graph_interrupt.py
git commit -m "feat(agent): узлы interpret/confirm с interrupt + checkpointer"
```

---

## Task 5: Runner — thread_id, resume, событие clarify

**Files:**
- Modify: `agent/runner.py`
- Create: `tests/test_runner_clarify.py`

- [ ] **Step 1: Написать падающий тест**

`tests/test_runner_clarify.py`:

```python
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import build_graph
from agent.runner import _astream_events
from tests.conftest import FakeChat


def _graph():
    return build_graph(
        plain_llm=FakeChat("погода 2026-06-02"),
        agent_llm=FakeChat(AIMessage("Готово")),
        checkpointer=MemorySaver(),
    )


async def test_start_emits_clarify_event():
    events = []
    async for ev in _astream_events("t1", query="погода", graph=_graph()):
        events.append(ev)
    clarify = [e for e in events if e["type"] == "clarify"]
    assert len(clarify) == 1
    assert clarify[0]["data"]["thread_id"] == "t1"
    assert clarify[0]["data"]["interpretation"] == "погода 2026-06-02"
    assert clarify[0]["data"]["round"] == 1


async def test_resume_confirmed_emits_final_and_done():
    graph = _graph()
    async for _ in _astream_events("t2", query="погода", graph=graph):
        pass
    events = []
    async for ev in _astream_events(
        "t2", resume={"confirmed": True, "correction": None}, graph=graph
    ):
        events.append(ev)
    types = [e["type"] for e in events]
    assert "final_answer" in types
    assert "done" in types
    final = next(e for e in events if e["type"] == "final_answer")
    assert final["data"]["text"] == "Готово"
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `.venv/bin/pytest tests/test_runner_clarify.py -v`
Expected: FAIL (TypeError: _astream_events() got an unexpected keyword argument 'query' / 'graph')

- [ ] **Step 3: Переписать `agent/runner.py`**

Полностью заменить содержимое `agent/runner.py` на:

```python
"""
Runner — обёртка над graph.astream(stream_mode="updates"), которая
конвертирует LangGraph-обновления в наши SSE-events.

Event types:
  - clarify      {thread_id, interpretation, original, round} — пауза подтверждения
  - node_start   {node}
  - tool_call    {name, args, id}
  - tool_result  {name, id, result}
  - final_answer {text}
  - done         {iterations}
  - error        {code, message}
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from agent.config import MAX_ITER
from agent.graph import build_graph


@dataclass
class AgentRun:
    answer: str
    trace: list[dict] = field(default_factory=list)
    iterations: int = 0


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _event(type_: str, data: dict) -> dict:
    return {"type": type_, "timestamp": _now_iso(), "data": data}


_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def _astream_events(
    thread_id: str,
    *,
    query: str | None = None,
    resume: dict | None = None,
    graph=None,
) -> AsyncIterator[dict]:
    """Async-генератор dict-events. Старт (query) или возобновление (resume).
    graph — для инъекции в тестах; по умолчанию singleton."""
    graph = graph if graph is not None else _get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    if resume is not None:
        stream_input = Command(resume=resume)
    else:
        stream_input = {"messages": [HumanMessage(query or "")]}

    iterations = 0
    stream = graph.astream(stream_input, config, stream_mode="updates")
    try:
        async for update in stream:
            # interrupt() — пауза подтверждения. Эмитим clarify и выходим:
            # граф сохранён в checkpointer, продолжим на resume.
            if "__interrupt__" in update:
                payload = update["__interrupt__"][0].value
                yield _event("clarify", {"thread_id": thread_id, **payload})
                return

            if iterations >= MAX_ITER:
                yield _event("error", {
                    "code": "max_iter",
                    "message": f"Превышен лимит итераций ({MAX_ITER})",
                })
                return
            iterations += 1

            for node_name, delta in update.items():
                yield _event("node_start", {"node": node_name})
                new_messages = (delta or {}).get("messages") or []
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
                            yield _event("final_answer", {"text": msg.content or ""})
                    elif isinstance(msg, ToolMessage):
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
        yield _event("error", {"code": type(exc).__name__, "message": str(exc)})


async def run_collect(query: str, thread_id: str = "collect") -> AgentRun:
    """Non-streaming прогон для POST /api/agent/ask. Интерактивного
    подтверждения нет — авто-принимаем первую интерпретацию."""
    trace: list[dict] = []
    answer = ""
    iterations = 0
    pending: dict | None = {"query": query}
    # Цикл: старт → если clarify, авто-resume confirmed=True → до done/error.
    while pending is not None:
        if "query" in pending:
            gen = _astream_events(thread_id, query=pending["query"])
        else:
            gen = _astream_events(thread_id, resume=pending["resume"])
        pending = None
        async for event in gen:
            trace.append(event)
            if event["type"] == "clarify":
                pending = {"resume": {"confirmed": True, "correction": None}}
            elif event["type"] == "final_answer":
                answer = event["data"]["text"]
            elif event["type"] == "done":
                iterations = event["data"]["iterations"]
            elif event["type"] == "error":
                answer = answer or f"Ошибка: {event['data']['message']}"
    return AgentRun(answer=answer, trace=trace, iterations=iterations)


async def run_stream(
    thread_id: str, *, query: str | None = None, resume: dict | None = None
) -> AsyncIterator[dict]:
    """Стрим events для GET-эндпоинтов (старт или resume)."""
    async for event in _astream_events(thread_id, query=query, resume=resume):
        yield event
```

- [ ] **Step 4: Запустить тесты runner**

Run: `.venv/bin/pytest tests/test_runner_clarify.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Прогнать все тесты**

Run: `.venv/bin/pytest -v`
Expected: PASS (все)

- [ ] **Step 6: Commit**

```bash
git add agent/runner.py tests/test_runner_clarify.py
git commit -m "feat(agent): runner — thread_id, resume, событие clarify"
```

---

## Task 6: Web — thread_id, эндпоинт resume, clarify, сохранение trace

**Files:**
- Modify: `web/server.py`

> Контекст: текущие эндпоинты — `agent_ask` (POST, строка 2027), `agent_ask_stream` (GET, строка 2051). `run_collect`/`run_stream` импортированы (строка 59). `_save_agent_message_async` — строка 2020. `_sse_event` — строка 1312.

- [ ] **Step 1: Обновить импорт runner**

В `web/server.py` строка 59 — убедиться, что импорт такой:

```python
from agent.runner import run_collect, run_stream
```

(уже есть; менять не нужно)

- [ ] **Step 2: Добавить uuid-импорт и буфер trace**

Рядом с прочими `import` сверху файла (после `import math`) добавить:

```python
import uuid
```

Перед эндпоинтом `agent_ask` (строка ~2027) добавить модульный буфер:

```python
# In-memory накопление trace по thread_id: clarify-круги и финальный ответ
# размазаны по нескольким HTTP-запросам одного turn'а. Парно с MemorySaver
# (тоже in-memory). На `done` сохраняем полный trace и чистим запись.
_agent_trace_buffers: dict[str, list[dict]] = {}
```

- [ ] **Step 3: Обновить POST /api/agent/ask (передать thread_id)**

Заменить тело `agent_ask` (строки ~2028-2043) на:

```python
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
```

- [ ] **Step 4: Переписать GET /api/agent/ask/stream (старт turn'а)**

Заменить `agent_ask_stream` (строки ~2051-2083) на:

```python
@app.get("/api/agent/ask/stream")
async def agent_ask_stream(query: str) -> StreamingResponse:
    """SSE-стрим старта turn'а. Генерит thread_id, стримит до первого clarify.
    user-сообщение сохраняется здесь; assistant — на событии done (в resume)."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="query пуст")

    thread_id = uuid.uuid4().hex
    await _save_agent_message_async("user", query, None)
    _agent_trace_buffers[thread_id] = []

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
```

- [ ] **Step 5: Добавить общий генератор и resume-эндпоинт**

Сразу после `agent_ask_stream` добавить:

```python
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
```

- [ ] **Step 6: Проверить, что сервер импортируется**

Run: `.venv/bin/python -c "import web.server; print('ok')"`
Expected: `ok` (возможны warning'и langchain — это норма)

- [ ] **Step 7: Ручная верификация эндпоинтов**

Поднять сервер: `.venv/bin/uvicorn web.server:app --port 8000` (в отдельном терминале).

Старт turn'а:

```bash
curl -N 'http://localhost:8000/api/agent/ask/stream?query=какая+погода+сегодня'
```

Expected: придёт `event: clarify` с `data`, содержащим `thread_id`, `interpretation` (с датой `2026-06-02`), `original`, `round: 1`. Стрим закрывается.

Скопировать `thread_id` из ответа и подтвердить:

```bash
curl -N 'http://localhost:8000/api/agent/resume/stream?thread_id=<TID>&confirmed=true'
```

Expected: пойдут `node_start`/`tool_call`/`tool_result`/`final_answer`/`done`.

Проверить «нет» (новый круг):

```bash
curl -N 'http://localhost:8000/api/agent/ask/stream?query=погода'   # взять новый TID
curl -N 'http://localhost:8000/api/agent/resume/stream?thread_id=<TID2>&confirmed=false&correction=я+про+завтра'
```

Expected: второй вызов вернёт новый `event: clarify` с `round: 2`.

Проверить, что в БД сохранён assistant с trace:

```bash
curl -s 'http://localhost:8000/api/agent/messages?limit=4'
```

Expected: последняя assistant-запись содержит `trace` с событиями `clarify` и `final_answer`.

- [ ] **Step 8: Commit**

```bash
git add web/server.py
git commit -m "feat(web): clarify-старт, resume-эндпоинт, накопление trace"
```

---

## Task 7: Фронт — типы и API-клиент

**Files:**
- Modify: `frontend/src/lib/agent-types.ts`
- Modify: `frontend/src/lib/agent-api.ts`

- [ ] **Step 1: Добавить тип события clarify**

В `frontend/src/lib/agent-types.ts` заменить `AgentEventType` и дописать интерфейс:

```typescript
export type AgentEventType =
  | "clarify"
  | "node_start"
  | "tool_call"
  | "tool_result"
  | "final_answer"
  | "done"
  | "error";

export interface ClarifyData {
  thread_id: string;
  interpretation: string;
  original: string;
  round: number;
}
```

- [ ] **Step 2: Добавить buildResumeUrl**

В `frontend/src/lib/agent-api.ts` внутри объекта `agentApi`, после `buildStreamUrl`, добавить:

```typescript
  /** Возобновление turn'а: подтверждение (да/нет + уточнение). GET для EventSource. */
  buildResumeUrl: (opts: { threadId: string; confirmed: boolean; correction?: string }) => {
    const params = new URLSearchParams({
      thread_id: opts.threadId,
      confirmed: String(opts.confirmed),
    });
    if (opts.correction) params.set("correction", opts.correction);
    return apiUrl(`/api/agent/resume/stream?${params.toString()}`);
  },
```

- [ ] **Step 3: Проверка типов**

Run: `cd frontend && npx tsc --noEmit`
Expected: без ошибок (или те же, что были до изменений)

- [ ] **Step 4: Commit**

```bash
git add frontend/src/lib/agent-types.ts frontend/src/lib/agent-api.ts
git commit -m "feat(frontend): тип события clarify и buildResumeUrl"
```

---

## Task 8: Фронт — store pendingClarify

**Files:**
- Modify: `frontend/src/stores/agent-ui.ts`

- [ ] **Step 1: Обновить импорт типов**

В `frontend/src/stores/agent-ui.ts` строку 10 заменить на:

```typescript
import type { TraceEvent, ClarifyData } from "@/lib/agent-types";
```

- [ ] **Step 2: Добавить поле в AgentDraft**

В интерфейс `AgentDraft` (строки 13-19) после `abort` добавить поле:

```typescript
  abort: (() => void) | null;
  pendingClarify: ClarifyData | null;  // карточка подтверждения (null — нет паузы)
```

(заменив существующую строку `abort: (() => void) | null;` на эти две)

- [ ] **Step 3: Добавить сеттер в интерфейс и реализацию**

В интерфейс `AgentUIState` (строки 21-27) после `setFinished` добавить:

```typescript
  setPendingClarify: (c: ClarifyData | null) => void;
```

В объект стора (строки 29-41) после `setFinished` добавить реализацию:

```typescript
  setPendingClarify: (c) =>
    set((s) => (s.draft ? { draft: { ...s.draft, pendingClarify: c } } : s)),
```

При создании нового `draft` в `agent-chat.tsx` поле `pendingClarify` инициализируется `null` (см. Task 10).

- [ ] **Step 4: Проверка типов**

Run: `cd frontend && npx tsc --noEmit`
Expected: появятся ошибки только в `agent-chat.tsx` (где `draft` создаётся без `pendingClarify`) — их закрывает Task 10. Если других ошибок нет — ок.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/stores/agent-ui.ts
git commit -m "feat(frontend): pendingClarify в agent-ui store"
```

---

## Task 9: Фронт — компонент ClarifyPrompt

**Files:**
- Create: `frontend/src/components/agent/clarify-prompt.tsx`

- [ ] **Step 1: Создать компонент**

`frontend/src/components/agent/clarify-prompt.tsx`:

```tsx
"use client";

import * as React from "react";
import { Check, X, HelpCircle, Send } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { ClarifyData } from "@/lib/agent-types";

export function ClarifyPrompt({
  data,
  onConfirm,
  onReject,
  disabled,
}: {
  data: ClarifyData;
  onConfirm: () => void;
  onReject: (correction: string) => void;
  disabled?: boolean;
}) {
  const [rejecting, setRejecting] = React.useState(false);
  const [correction, setCorrection] = React.useState("");

  const submitReject = () => {
    const c = correction.trim();
    if (!c) return;
    onReject(c);
    setRejecting(false);
    setCorrection("");
  };

  return (
    <div className="rounded-2xl border border-amber-400/40 bg-amber-50/50 dark:bg-amber-950/20 px-4 py-3 space-y-3">
      <div className="flex items-start gap-2">
        <HelpCircle className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
        <div className="text-sm">
          <span className="text-muted-foreground">Вы имели в виду:</span>{" "}
          <span className="font-medium">«{data.interpretation}»</span>
        </div>
      </div>

      {!rejecting ? (
        <div className="flex gap-2">
          <Button size="sm" onClick={onConfirm} disabled={disabled} className="gap-1">
            <Check className="h-3.5 w-3.5" /> Да, верно
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setRejecting(true)}
            disabled={disabled}
            className="gap-1"
          >
            <X className="h-3.5 w-3.5" /> Нет
          </Button>
        </div>
      ) : (
        <div className="flex items-end gap-2">
          <Textarea
            autoFocus
            value={correction}
            onChange={(e) => setCorrection(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submitReject();
              }
            }}
            placeholder="Что вы имели в виду?"
            className="flex-1 min-h-[38px] max-h-[120px] resize-none"
            rows={1}
          />
          <Button size="icon" onClick={submitReject} disabled={disabled || !correction.trim()} className="h-9 w-9 shrink-0">
            <Send className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Проверка типов**

Run: `cd frontend && npx tsc --noEmit`
Expected: без новых ошибок в этом файле (ошибки в agent-chat.tsx ещё остаются до Task 10)

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/agent/clarify-prompt.tsx
git commit -m "feat(frontend): компонент ClarifyPrompt"
```

---

## Task 10: Фронт — openStream и интеграция clarify в AgentChat

**Files:**
- Modify: `frontend/src/components/agent/agent-chat.tsx`

- [ ] **Step 1: Добавить импорты и сеттер**

В импортах `agent-chat.tsx` добавить:

```tsx
import { agentApi } from "@/lib/agent-api";
import type { AgentMessage, TraceEvent, ClarifyData } from "@/lib/agent-types";
import { ClarifyPrompt } from "./clarify-prompt";
```

(заменив существующую строку импорта типов на вариант с `ClarifyData`)

В деструктуризацию стора добавить:

```tsx
  const setPendingClarify = useAgentUi((s) => s.setPendingClarify);
```

- [ ] **Step 2: Вынести openStream и переписать send**

Заменить `send` (строки ~37-86) на функцию `openStream` + новый `send`:

```tsx
  const openStream = React.useCallback(
    (url: string) => {
      const es = new EventSource(url);
      const abort = () => es.close();
      // Обновляем только abort, остальной draft не трогаем.
      useAgentUi.setState((s) => (s.draft ? { draft: { ...s.draft, abort } } : {}));

      const onEvent = (type: TraceEvent["type"]) => (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          const ev: TraceEvent = { type, timestamp: new Date().toISOString(), data };
          appendTrace(ev);

          if (type === "clarify") {
            setPendingClarify(data as ClarifyData);
            es.close(); // граф на паузе; продолжим через resume
            return;
          }
          if (type === "final_answer" && typeof data.text === "string") {
            setAnswer(data.text);
          }
          if (type === "done" || type === "error") {
            setFinished(true);
            es.close();
            queryClient.invalidateQueries({ queryKey: ["agent", "messages"] });
            setTimeout(() => setDraft(null), 800);
          }
        } catch (err) {
          console.error("SSE parse failed", err);
        }
      };

      es.addEventListener("clarify", onEvent("clarify"));
      es.addEventListener("node_start", onEvent("node_start"));
      es.addEventListener("tool_call", onEvent("tool_call"));
      es.addEventListener("tool_result", onEvent("tool_result"));
      es.addEventListener("final_answer", onEvent("final_answer"));
      es.addEventListener("done", onEvent("done"));
      es.addEventListener("error", (e) => {
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
    },
    [appendTrace, setAnswer, setFinished, setDraft, setPendingClarify, queryClient],
  );

  const send = React.useCallback(() => {
    const q = text.trim();
    if (!q || draft) return;
    setText("");
    setDraft({
      userQuery: q,
      trace: [],
      answer: "",
      finished: false,
      abort: null,
      pendingClarify: null,
    });
    openStream(agentApi.buildStreamUrl(q));
  }, [text, draft, setDraft, openStream]);

  const onConfirm = React.useCallback(() => {
    const pc = draft?.pendingClarify;
    if (!pc) return;
    setPendingClarify(null);
    openStream(agentApi.buildResumeUrl({ threadId: pc.thread_id, confirmed: true }));
  }, [draft, setPendingClarify, openStream]);

  const onReject = React.useCallback(
    (correction: string) => {
      const pc = draft?.pendingClarify;
      if (!pc) return;
      setPendingClarify(null);
      openStream(
        agentApi.buildResumeUrl({ threadId: pc.thread_id, confirmed: false, correction }),
      );
    },
    [draft, setPendingClarify, openStream],
  );
```

> Примечание: если поле `draft` создаётся со строгим типом и требует все поля — `pendingClarify: null` уже добавлено выше. Сверить с типом из Task 8.

- [ ] **Step 3: Отрисовать карточку под draft-пузырём**

В JSX, в блоке `{draft && (...)}` (строки ~129-139), после assistant-`Bubble` добавить:

```tsx
              {draft.pendingClarify && (
                <div className="ml-10">
                  <ClarifyPrompt
                    data={draft.pendingClarify}
                    onConfirm={onConfirm}
                    onReject={onReject}
                  />
                </div>
              )}
```

- [ ] **Step 4: Учесть pendingClarify в isStreaming**

Найти строку `const isStreaming = draft !== null && !draft.finished;` и заменить на:

```tsx
  // Во время ожидания подтверждения не показываем «стоп»/курсор стрима.
  const isStreaming = draft !== null && !draft.finished && !draft.pendingClarify;
```

- [ ] **Step 5: Проверка типов и сборка**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: без ошибок

- [ ] **Step 6: Ручная верификация в UI**

Поднять бэк (`uvicorn web.server:app --port 8000`) и фронт (`cd frontend && npm run dev`), открыть `/agent`.

1. Спросить «какая погода сегодня» → под «Думаю…» появляется карточка «Вы имели в виду: «…2026-06-02…?»» с кнопками **Да, верно** / **Нет**.
2. Нажать **Да, верно** → карточка исчезает, идут шаги агента и финальный ответ.
3. Новый запрос → **Нет** → появляется поле → ввести «я про завтра» → Enter → карточка обновляется новой интерпретацией (round 2).
4. Перезагрузить страницу → в истории assistant-сообщение содержит trace (TraceTimeline отрисуется после Task 11).

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/agent/agent-chat.tsx
git commit -m "feat(frontend): openStream + карточка подтверждения clarify"
```

---

## Task 11: Фронт — отрисовка шага clarify в TraceTimeline

**Files:**
- Modify: `frontend/src/components/agent/trace-step.tsx`

> Контекст: компонент рисует шаг тремя `switch`-функциями по `type`: `eventIcon` (строки 12-21), `eventTitle` (23-48), `eventColor` (50-58). Поскольку Task 7 добавил `"clarify"` в union `AgentEventType`, ветку нужно добавить во ВСЕ три — иначе `eventIcon` вернёт `undefined` и `<Icon />` упадёт.

- [ ] **Step 1: Добавить иконку в eventIcon**

Импортировать `HelpCircle` — в строке 4 заменить импорт на:

```tsx
import { ChevronRight, Wrench, MessageSquare, Brain, CheckCircle2, AlertCircle, HelpCircle } from "lucide-react";
```

В `eventIcon` (строки 12-21) перед `case "node_start"` добавить:

```tsx
    case "clarify":      return HelpCircle;
```

- [ ] **Step 2: Добавить заголовок в eventTitle**

В `eventTitle` (строки 23-48), внутри `switch (ev.type)`, перед `case "node_start"` добавить:

```tsx
    case "clarify": {
      const interp = ev.data.interpretation as string;
      const round = (ev.data.round as number) ?? 0;
      return `🤔 Уточнение${round > 1 ? ` (круг ${round})` : ""}: «${interp}»`;
    }
```

- [ ] **Step 3: Добавить цвет в eventColor**

В `eventColor` (строки 50-58), перед `default:` добавить:

```tsx
    case "clarify":       return "text-amber-400";
```

- [ ] **Step 5: Проверка типов и сборка**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: без ошибок

- [ ] **Step 6: Ручная верификация**

В UI пройти turn с одним «нет» и одним «да», затем раскрыть TraceTimeline у ответа — в нём видны шаги `🤔 Уточнение: «…»` (в т.ч. «круг 2»).

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/agent/trace-step.tsx
git commit -m "feat(frontend): отрисовка шага clarify в TraceTimeline"
```

---

## Финальная проверка

- [ ] **Все Python-тесты зелёные**

Run: `.venv/bin/pytest -v`
Expected: PASS (test_clarify: 3, test_graph_interrupt: 4, test_runner_clarify: 2)

- [ ] **Фронт собирается**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: без ошибок

- [ ] **End-to-end дымовой прогон** (бэк + фронт подняты)

Пройти сценарий: вопрос «какая погода сегодня» → карточка с датой → «Нет» → уточнение → «Да» → агент дёргает `get_weather` на подтверждённую дату → финальный ответ. История переживает перезагрузку, trace содержит шаги clarify.

---

## Покрытие спеки (self-review)

- Узлы interpret/confirm, форма графа → Task 2, 4.
- State-поля → Task 1.
- MAX_CLARIFY_ROUNDS → Task 3, проверка в Task 4.
- MemorySaver + thread_id → Task 4, 5, 6.
- Событие clarify + `__interrupt__` → Task 5.
- Эндпоинты ask/stream (старт) и resume/stream → Task 6.
- Накопление trace по thread_id + сохранение на done → Task 6.
- run_collect авто-подтверждение (POST не интерактивен) → Task 5.
- agent_node использует effective_query → Task 4.
- Типы/API/store/компонент/openStream/trace clarify → Task 7-11.
- Вне скоупа (PostgresSaver, «только при неоднозначности», таймаут) — не реализуется, согласовано.
</content>
