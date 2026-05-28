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


def _extract_reservations(raw) -> list:
    """skkrondo может вернуть либо list, либо {"items": [...], "total": N}.
    Приводим к плоскому list для downstream-кода."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("items", "reservations", "data", "results"):
            v = raw.get(key)
            if isinstance(v, list):
                return v
    return []


def _summarize_courts(raw: list) -> str:
    """Сжимаем массив броней в одну строку — чтобы не раздувать LLM-контекст.
    На элементах-не-dict молча пропускаем — лучше неполный summary, чем 500."""
    if not raw:
        return "Все корты свободны."
    by_court: dict[object, list[str]] = {}
    for r in raw:
        if not isinstance(r, dict):
            continue
        court = r.get("court_id") or r.get("court", "?")
        slot = f"{r.get('start_time', '?')}-{r.get('end_time', '?')}"
        by_court.setdefault(court, []).append(slot)
    if not by_court:
        return "Все корты свободны."
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
    reservations = _extract_reservations(raw)
    return {
        "date": date,
        "reservations": reservations,
        "summary": _summarize_courts(reservations),
    }


TOOLS = [get_weather, get_courts_availability]
