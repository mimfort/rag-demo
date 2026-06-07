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


# Условия в weather-API, которые мы считаем «солнечно». Финальная
# интерпретация на LLM — это просто подсказка-флаг.
_SUNNY_KEYWORDS = (
    "ясно", "безоблачно", "малооблачно", "солнечно", "clear", "sunny",
)


def _classify_sunny(condition) -> bool:
    """Эвристика по текстовому описанию погоды — солнечно или нет."""
    text = str(condition or "").lower()
    return any(kw in text for kw in _SUNNY_KEYWORDS)


def _pick_midday(hours: list) -> dict | None:
    """Из почасового прогноза дня берём точку, ближайшую к полудню (12:00 UTC).

    `/gismeteo/forecast/{day}` отдаёт сводку (`summary`) + почасовку (`hours`,
    время в UTC). Дневную температуру берём с точки около полудня — это ближе
    к ощущаемой днём, чем min/max за сутки. None, если почасовки нет.
    """
    pts = [
        h for h in hours
        if isinstance(h, dict) and h.get("temperature_c") is not None
    ]
    if not pts:
        return None

    def _hour(h: dict) -> int:
        try:
            return int(str(h.get("time", ""))[11:13])
        except ValueError:
            return 0

    return min(pts, key=lambda h: abs(_hour(h) - 12))


def _court_free_hours(court: dict) -> list[str]:
    """Свободные часы одного корта как ['10:00', '11:00', ...].

    Эндпоинт зовётся с ?status=free, поэтому в slots приходят только свободные
    часы. На всякий случай (defense-in-depth) фильтруем по status == "free".
    """
    hours: list[str] = []
    for s in court.get("slots") or []:
        if not isinstance(s, dict):
            continue
        if s.get("status") not in (None, "free"):
            continue
        h = s.get("hour")
        try:
            hours.append(f"{int(h):02d}:00")
        except (TypeError, ValueError):
            continue
    return hours


def _summarize_free(courts: list) -> str:
    """Сжимаем СВОБОДНЫЕ слоты кортов в одну строку для LLM-контекста.

    Показываем только доступное — модель не должна выдумывать занятость.
    На элементах-не-dict молча пропускаем — лучше неполный summary, чем 500.
    """
    if not courts:
        return "Нет данных о свободных слотах на эту дату."
    parts: list[str] = []
    any_free = False
    for c in courts:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or f"корт {c.get('court_id', '?')}").strip()
        hours = _court_free_hours(c)
        if hours:
            any_free = True
            parts.append(f"{name}: свободно {', '.join(hours)}")
        else:
            parts.append(f"{name}: свободных слотов нет")
    if not parts:
        return "Нет данных о свободных слотах на эту дату."
    if not any_free:
        return "Свободных слотов нет — все корты заняты."
    return "; ".join(parts)


@tool
async def get_weather(date: str) -> dict:
    """Получить прогноз погоды на конкретную дату в формате YYYY-MM-DD.

    Возвращает: {date, temperature_c, temp_min_c, temp_max_c, condition,
    precipitation_mm, sunny: bool} либо {error: "..."} при сетевой/HTTP
    ошибке, невалидной дате или отсутствии данных на запрошенный день.
    """
    valid = _validate_date(date)
    if valid is None:
        return {"error": "invalid date: expected YYYY-MM-DD", "date": date}
    # /gismeteo/forecast/{day} отдаёт по дню сводку (summary: min/max/описание)
    # + почасовку (hours, UTC). Прогноз обычно на несколько дней вперёд.
    url = (
        f"{agent_settings.skkrondo_base_url}"
        f"/gismeteo/forecast/{quote(valid, safe='')}"
    )
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
            r = await client.get(url)
            r.raise_for_status()
            raw = r.json()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "date": valid}
    summary = raw.get("summary") if isinstance(raw, dict) else None
    if not isinstance(summary, dict):
        return {
            "error": "no forecast for this date (прогноз ограничен по горизонту)",
            "date": valid,
        }
    hours = raw.get("hours") if isinstance(raw, dict) else None
    midday = _pick_midday(hours or [])
    condition = summary.get("description")
    return {
        "date": valid,
        "temperature_c": midday.get("temperature_c") if midday else None,
        "temp_min_c": summary.get("temp_min_c"),
        "temp_max_c": summary.get("temp_max_c"),
        "condition": condition,
        "precipitation_mm": summary.get("precipitation_mm"),
        "sunny": _classify_sunny(condition),
    }


@tool
async def get_courts_availability(date: str) -> dict:
    """Получить СВОБОДНЫЕ слоты кортов на конкретную дату YYYY-MM-DD.

    Возвращает только доступные для брони часы (занятые не показываются) —
    рекомендуй пользователю исключительно эти слоты, ничего не выдумывай.

    Структура: {date, open_hour, close_hour, courts: [{court_id, name,
    free_hours: ["10:00", ...]}], summary: "текст"} либо {error: "..."}
    при сетевой/HTTP ошибке или невалидной дате.
    """
    valid = _validate_date(date)
    if valid is None:
        return {"error": "invalid date: expected YYYY-MM-DD", "date": date}
    # ?status=free — API возвращает только свободные слоты, чтобы LLM физически
    # не могла увидеть/перепутать занятые часы и навыдумывать.
    url = (
        f"{agent_settings.skkrondo_base_url}"
        f"/court_reservations/availability/{quote(valid, safe='')}?status=free"
    )
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
            r = await client.get(url)
            r.raise_for_status()
            raw = r.json()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "date": valid}
    courts_raw = raw.get("courts") if isinstance(raw, dict) else None
    courts_raw = courts_raw if isinstance(courts_raw, list) else []
    courts = [
        {
            "court_id": c.get("court_id"),
            "name": str(c.get("name") or "").strip() or None,
            "free_hours": _court_free_hours(c),
        }
        for c in courts_raw
        if isinstance(c, dict)
    ]
    return {
        "date": valid,
        "open_hour": raw.get("open_hour") if isinstance(raw, dict) else None,
        "close_hour": raw.get("close_hour") if isinstance(raw, dict) else None,
        "courts": courts,
        "summary": _summarize_free(courts_raw),
    }


TOOLS = [get_weather, get_courts_availability]
