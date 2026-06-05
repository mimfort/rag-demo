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

from datetime import date as date_cls
from urllib.parse import quote

import httpx
from langchain_core.tools import tool

from agent.config import HTTP_TIMEOUT_SEC, agent_settings


def _validate_date(value: str) -> str | None:
    """Возвращает дату как YYYY-MM-DD если валидна, иначе None.

    Защита от SSRF/path-injection: `date` приходит из tool_call'а LLM, и
    без проверки попадёт в URL-интерполяцию. Маленькая модель теоретически
    может сгенерировать аргумент вроде `../../admin`. fromisoformat()
    раскладывает только настоящие ISO-даты — невалидные кидают ValueError.
    Дополнительно квотируем сегмент для defense-in-depth.
    """
    try:
        parsed = date_cls.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    return parsed.isoformat()


# Условия в weather-API, которые мы считаем «солнечно». На реальной
# структуре ответа skkrondo может быть лучше уточнить (см. /weather/weather/{date}).
_SUNNY_KEYWORDS = (
    "ясно", "безоблачно", "малооблачно", "солнечно", "clear", "sunny",
)


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
        # У брони skkrondo один час начала (поле `time`, целое), без end_time —
        # слоты часовые. Старый код читал start_time/end_time → выдавал "?-?".
        t = r.get("time")
        try:
            slot = f"{int(t):02d}:00"
        except (TypeError, ValueError):
            slot = "?"
        by_court.setdefault(court, []).append(slot)
    if not by_court:
        return "Все корты свободны."
    parts = [
        f"корт {c}: занят {', '.join(slots)}"
        for c, slots in sorted(by_court.items(), key=lambda x: str(x[0]))
    ]
    return "; ".join(parts)


def _pick_day_forecast(forecast: list, valid: str) -> dict | None:
    """Из почасового gismeteo-прогноза выбираем сводку по нужному дню.

    gismeteo отдаёт весь массив сразу (без параметра date), время в UTC.
    Берём все точки запрошенной даты, считаем диапазон температур и в
    качестве «condition» — точку, ближайшую к полудню (12:00 UTC ≈ день).
    None, если на эту дату данных нет (прогноз обычно ~4 дня вперёд).
    """
    day = [
        e for e in forecast
        if isinstance(e, dict) and str(e.get("date", ""))[:10] == valid
    ]
    if not day:
        return None
    temps = [e["temperature_C"] for e in day if e.get("temperature_C") is not None]

    def _hour(e: dict) -> int:
        try:
            return int(str(e.get("date", ""))[11:13])
        except ValueError:
            return 0

    midday = min(day, key=lambda e: abs(_hour(e) - 12))
    return {
        "temperature_c": midday.get("temperature_C"),
        "temp_min_c": min(temps) if temps else None,
        "temp_max_c": max(temps) if temps else None,
        "condition": midday.get("description"),
        "midday": midday,
    }


@tool
async def get_weather(date: str) -> dict:
    """Получить прогноз погоды на конкретную дату в формате YYYY-MM-DD.

    Возвращает: {date, temperature_c, temp_min_c, temp_max_c, condition,
    sunny: bool, raw} либо {error: "..."} при сетевой/HTTP ошибке,
    невалидной дате или отсутствии данных на запрошенный день.
    """
    valid = _validate_date(date)
    if valid is None:
        return {"error": "invalid date: expected YYYY-MM-DD", "date": date}
    # Эндпоинт /weather/weather/{date} стабильно отдаёт 500 (баг на стороне
    # skkrondo), поэтому используем рабочий gismeteo: почасовой прогноз на
    # ~4 дня вперёд, фильтруем по нужной дате на нашей стороне.
    url = f"{agent_settings.skkrondo_base_url}/gismeteo/weather_gismeteo4/"
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
            r = await client.get(url)
            r.raise_for_status()
            raw = r.json()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "date": valid}
    forecast = raw.get("forecast") if isinstance(raw, dict) else None
    summary = _pick_day_forecast(forecast or [], valid)
    if summary is None:
        return {
            "error": "no forecast for this date (gismeteo даёт ~4 дня вперёд)",
            "date": valid,
        }
    return {
        "date": valid,
        "temperature_c": summary["temperature_c"],
        "temp_min_c": summary["temp_min_c"],
        "temp_max_c": summary["temp_max_c"],
        "condition": summary["condition"],
        "sunny": _classify_sunny(summary["midday"]),
        "raw": summary["midday"],
    }


@tool
async def get_courts_availability(date: str) -> dict:
    """Получить занятые слоты кортов на конкретную дату YYYY-MM-DD.

    Возвращает: {date, reservations: [...], summary: "текст"} либо
    {error: "..."} при сетевой/HTTP ошибке или невалидной дате.
    """
    valid = _validate_date(date)
    if valid is None:
        return {"error": "invalid date: expected YYYY-MM-DD", "date": date}
    url = f"{agent_settings.skkrondo_base_url}/court_reservations/all/{quote(valid, safe='')}"
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
            r = await client.get(url)
            r.raise_for_status()
            raw = r.json()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "date": valid}
    reservations = _extract_reservations(raw)
    return {
        "date": valid,
        "reservations": reservations,
        "summary": _summarize_courts(reservations),
    }


TOOLS = [get_weather, get_courts_availability]
