"""
pricing.py — ценовой гейт для chat-модели (OpenRouter).

OpenRouter отдаёт цены моделей через GET {base}/models: у каждого элемента
data[] есть id и pricing.{prompt,completion} — строки «USD за 1 токен».
Умножаем на 1e6 → цена за 1М токенов и сравниваем с лимитами из .env.

Гейт идемпотентный (одна проверка на процесс) и fail-open: если цену нельзя
узнать (сеть недоступна, модели нет в списке, не-OpenRouter base_url) — пишем
warning и продолжаем, а не блокируем работу.

Смена провайдера: гейт активен только когда заданы лимиты; на пустых лимитах
не делает ни одного сетевого вызова.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class ModelPrice:
    """Цена модели в USD за 1М токенов."""

    prompt_per_mtok: float
    completion_per_mtok: float


class PriceCapExceeded(RuntimeError):
    """Цена модели выше заданного в .env лимита — работа прерывается."""


# Идемпотентность: проверяем цену один раз за процесс.
_already_enforced = False


def _to_mtok(raw: object) -> float:
    """Строка/число «USD за токен» → float «USD за 1М токенов»."""
    try:
        return float(raw or 0.0) * 1_000_000
    except (TypeError, ValueError):
        return 0.0


def fetch_prices(
    base_url: str,
    api_key: str,
    *,
    timeout: float = 10.0,
    client: httpx.Client | None = None,
) -> dict[str, ModelPrice]:
    """Возвращает карту id → ModelPrice по данным GET {base}/models."""
    owns_client = client is None
    client = client or httpx.Client(
        timeout=timeout,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        response = client.get(f"{base_url.rstrip('/')}/models")
        response.raise_for_status()
        data = response.json().get("data", [])
    finally:
        if owns_client:
            client.close()

    prices: dict[str, ModelPrice] = {}
    for item in data:
        mid = item.get("id")
        pricing = item.get("pricing") or {}
        if not mid:
            continue
        prices[mid] = ModelPrice(
            prompt_per_mtok=_to_mtok(pricing.get("prompt")),
            completion_per_mtok=_to_mtok(pricing.get("completion")),
        )
    return prices


def enforce_price_caps_once(settings, *, client: httpx.Client | None = None) -> None:
    """
    Один раз за процесс проверяет цену основной и запасной модели против
    лимитов. Превышение → PriceCapExceeded. Недоступный прайсинг / отсутствие
    модели в /models → warning и продолжаем (fail-open).
    """
    global _already_enforced
    if _already_enforced:
        return

    max_prompt = settings.llm_max_prompt_price
    max_completion = settings.llm_max_completion_price
    if max_prompt is None and max_completion is None:
        _already_enforced = True
        return

    models = [settings.llm_model]
    if settings.llm_fallback_model:
        models.append(settings.llm_fallback_model)

    try:
        prices = fetch_prices(settings.llm_base_url, settings.llm_api_key, client=client)
    except Exception as exc:  # noqa: BLE001 — fail-open by design
        print(f"⚠ Не удалось получить цены моделей ({exc}); ценовой гейт пропущен.")
        _already_enforced = True
        return

    for model in models:
        price = prices.get(model)
        if price is None:
            print(
                f"⚠ Модель '{model}' не найдена в /models — цену проверить нельзя, "
                f"пропускаю (fail-open)."
            )
            continue
        if max_prompt is not None and price.prompt_per_mtok > max_prompt:
            raise PriceCapExceeded(
                f"Модель '{model}': prompt ${price.prompt_per_mtok:.2f}/1М > "
                f"лимита ${max_prompt:.2f}/1М. Подними LLM_MAX_PROMPT_PRICE_PER_MTOK "
                f"или выбери модель дешевле."
            )
        if max_completion is not None and price.completion_per_mtok > max_completion:
            raise PriceCapExceeded(
                f"Модель '{model}': completion ${price.completion_per_mtok:.2f}/1М > "
                f"лимита ${max_completion:.2f}/1М. Подними "
                f"LLM_MAX_COMPLETION_PRICE_PER_MTOK или выбери модель дешевле."
            )

    _already_enforced = True
