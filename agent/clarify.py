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
будет работать агент (он умеет получать погоду и свободные слоты кортов на конкретные даты).

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
