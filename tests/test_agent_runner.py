"""
Тесты runner'а агента — конвертация LangGraph-обновлений в SSE-events.

Регрессия: пустой AIMessage (модель иногда возвращает пустой content на
verify-доработке) НЕ должен эмититься как final_answer — иначе он перезатирал
бы предыдущий нормальный ответ, и в UI/БД уходила бы пустота.
"""

import asyncio

from langchain_core.messages import AIMessage

from agent.runner import _astream_events


class _FakeGraph:
    """Граф-заглушка: astream возвращает заранее заданную серию updates."""

    def __init__(self, updates: list[dict]) -> None:
        self._updates = updates

    def astream(self, stream_input, config, stream_mode=None):
        updates = self._updates

        async def gen():
            for u in updates:
                yield u

        return gen()


def _collect(updates: list[dict]) -> list[dict]:
    async def run():
        events = []
        async for ev in _astream_events("t", query="x", graph=_FakeGraph(updates)):
            events.append(ev)
        return events

    return asyncio.run(run())


def test_empty_rewrite_final_answer_not_emitted():
    # agent даёт хороший ответ → verify находит расхождение → agent на
    # доработке возвращает пустой content → verify ok → done.
    updates = [
        {"agent": {"messages": [AIMessage(content="ХОРОШИЙ ОТВЕТ")]}},
        {"verify": {"verify_ok": False, "verify_issue": "расхождение"}},
        {"agent": {"messages": [AIMessage(content="")]}},
        {"verify": {"verify_ok": True, "verify_issue": None}},
    ]
    events = _collect(updates)
    finals = [e["data"]["text"] for e in events if e["type"] == "final_answer"]
    # Пустой rewrite не эмитится — последним финальным ответом остаётся хороший.
    assert finals == ["ХОРОШИЙ ОТВЕТ"]


def test_whitespace_only_final_answer_not_emitted():
    updates = [
        {"agent": {"messages": [AIMessage(content="   \n  ")]}},
    ]
    events = _collect(updates)
    finals = [e for e in events if e["type"] == "final_answer"]
    assert finals == []


def test_normal_final_answer_still_emitted():
    updates = [
        {"agent": {"messages": [AIMessage(content="Ответ по существу")]}},
    ]
    events = _collect(updates)
    finals = [e["data"]["text"] for e in events if e["type"] == "final_answer"]
    assert finals == ["Ответ по существу"]
