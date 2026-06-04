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
    # Ранний фидбек. interpret_node делает медленный LLM-вызов, а при
    # stream_mode="updates" его node_start пришёл бы только ПОСЛЕ завершения
    # (~25-45с на медленной локальной модели) — всё это время UI выглядит
    # зависшим. Эмитим node_start заранее, чтобы прогресс был виден сразу.
    # Только на старте turn'а: на resume первым идёт confirm/agent, не interpret.
    pre_announced_interpret = resume is None
    if pre_announced_interpret:
        yield _event("node_start", {"node": "interpret"})

    stream = graph.astream(stream_input, config, stream_mode="updates")
    try:
        async for update in stream:
            # interrupt() — пауза подтверждения. Эмитим clarify и выходим:
            # граф сохранён в checkpointer, продолжим на resume.
            if "__interrupt__" in update:
                payload = update["__interrupt__"][0].value
                yield _event("clarify", {"thread_id": thread_id, **payload})
                return

            # Лимит и счётчик «итераций» относятся к ReAct-циклу (agent/tools),
            # а не к clarify-фазе (interpret/confirm) — иначе «Готово (итераций: N)»
            # завышается, а бюджет MAX_ITER съедается узлами подтверждения.
            is_react = any(n in ("agent", "tools") for n in update)
            if is_react:
                if iterations >= MAX_ITER:
                    yield _event("error", {
                        "code": "max_iter",
                        "message": f"Превышен лимит итераций ({MAX_ITER})",
                    })
                    return
                iterations += 1

            for node_name, delta in update.items():
                # interpret уже анонсировали заранее (ранний фидбек) — не
                # дублируем его шаг. У interpret нет messages, так что
                # пропускаем весь блок узла.
                if node_name == "interpret" and pre_announced_interpret:
                    pre_announced_interpret = False
                    continue
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
    finally:
        # Детерминированно закрываем подлежащий astream. Без этого ранний
        # `return` (clarify-пауза, max_iter) оставляет недослитый генератор
        # и pending callback-таски LangChain — в долгоживущем сервере это
        # копилось бы на каждый turn с подтверждением.
        await stream.aclose()


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
