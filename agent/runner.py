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


# Модульный singleton — graph stateless (state per-invocation в astream),
# а build_graph() тяжёлый: создаёт ChatOpenAI, биндит tools, компилирует
# StateGraph. Пересоздавать на каждый запрос — лишний оверхед.
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def _astream_events(query: str) -> AsyncIterator[dict]:
    """Внутренний async-генератор: yield'ит dict-events по мере
    выполнения графа. НЕ форматирует SSE — это делает Web-слой."""
    graph = _get_graph()
    iterations = 0
    stream = graph.astream(
        {"messages": [HumanMessage(query)]},
        stream_mode="updates",
    )

    try:
        # stream_mode="updates" даёт {node_name: state_delta} после каждого узла.
        async for update in stream:
            # Guard ДО обработки очередного апдейта — иначе MAX_ITER+1-я
            # итерация уже выполнилась внутри графа когда мы её ловим.
            if iterations >= MAX_ITER:
                yield _event("error", {
                    "code": "max_iter",
                    "message": f"Превышен лимит итераций ({MAX_ITER})",
                })
                return
            iterations += 1

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
