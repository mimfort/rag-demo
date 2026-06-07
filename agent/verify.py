"""
Шаг верификации финального ответа (anti-hallucination).

Отдельный LLM-вызов сверяет финальный ответ агента с результатами tool'ов
(ToolMessage), полученными в этом turn'е. Локальная модель склонна привирать
числа — температуру, диапазон часов слотов, даты; verify ловит расхождения и
возвращает агента на доработку.

Контракт LLM намеренно простой (а не JSON): слабая локальная модель надёжнее
отдаёт «OK» / «ПРОБЛЕМА: …», чем валидный JSON. Вынесено отдельным модулем —
чтобы покрыть unit-тестами без сборки всего графа (как clarify.py).
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage


VERIFIER_PROMPT = """Ты — контролёр ответов помощника СКК «Рондо». Тебе дают: \
запрос пользователя, СЫРЫЕ данные из инструментов (погода, занятость кортов) и \
финальный ответ помощника. Задача — поймать только ГРУБЫЕ фактические искажения, \
не придираясь к мелочам.

Как читать данные о кортах: ориентируйся на поле "summary" и "free_hours" — \
это перечень СВОБОДНЫХ часов по кортам. Инструмент отдаёт ТОЛЬКО свободные \
слоты; занятых часов в данных нет вообще. Поэтому любые «занятые» часы в \
ответе помощника проверить нельзя — но если он называет час свободным, этого \
часа НЕ должно быть в free_hours/summary как-то иначе.

Помечай ПРОБЛЕМА только при ОЧЕВИДНОМ противоречии данным:
- выдуманная/искажённая температура, погода или дата;
- ответ называет час свободным, хотя его нет в free_hours/summary;
- ответ перечисляет конкретные «занятые» часы (их нет в данных — это выдумка);
- ответ ссылается на дату, которой нет в данных.

НЕ считай проблемой: краткость и стиль, неполный перечень часов, отсутствие \
деталей, формулировки, округление температуры. Если сомневаешься — пиши OK.

Ответь СТРОГО одним из двух и ничего больше:
- «OK» — если грубых искажений нет.
- «ПРОБЛЕМА: <кратко суть противоречия>» — только при явном расхождении."""


def _format_tool_results(tool_results: list[dict]) -> str:
    """Собирает результаты инструментов в компактный текст для контролёра."""
    if not tool_results:
        return "(инструменты не вызывались)"
    lines = []
    for r in tool_results:
        name = r.get("name") or "tool"
        content = r.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        lines.append(f"[{name}] {content}")
    return "\n".join(lines)


def build_verifier_messages(
    query: str, tool_results: list[dict], answer: str
) -> list:
    """Собирает messages для LLM-контролёра."""
    system = SystemMessage(VERIFIER_PROMPT)
    human = HumanMessage(
        f"Запрос пользователя: {query}\n\n"
        f"Данные из инструментов:\n{_format_tool_results(tool_results)}\n\n"
        f"Ответ помощника:\n{answer}\n\n"
        f"Проверь ответ."
    )
    return [system, human]


def parse_verdict(text: str) -> tuple[bool, str]:
    """Разбирает ответ контролёра в (ok, issue)."""
    text = (text or "").strip()
    if text.upper().startswith("OK"):
        return True, ""
    issue = text
    for prefix in ("ПРОБЛЕМА:", "ПРОБЛЕМА", "PROBLEM:", "PROBLEM"):
        if issue.upper().startswith(prefix.upper()):
            issue = issue[len(prefix):].lstrip(" :").strip()
            break
    return False, issue or "ответ не подтверждается данными инструментов"


async def verify_answer(
    llm, query: str, tool_results: list[dict], answer: str
) -> tuple[bool, str]:
    """Зовёт LLM-контролёра и возвращает (ok, issue)."""
    messages = build_verifier_messages(query, tool_results, answer)
    response = await llm.ainvoke(messages)
    return parse_verdict(response.content or "")
