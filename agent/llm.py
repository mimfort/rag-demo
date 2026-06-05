"""
Фабрика LLM. ChatOpenAI работает с любым OpenAI-совместимым провайдером
(по умолчанию OpenRouter): отличается только base_url/ключ/модель в .env.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from rag.config import settings as rag_settings
from rag.generator import build_routing_fields
from rag.pricing import enforce_price_caps_once


def make_llm() -> ChatOpenAI:
    """
    Возвращает ChatOpenAI, настроенную на chat-провайдера из .env
    (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).

    Перед созданием клиента проверяем ценовой лимит модели (см. pricing.py).
    Если заданы запасная модель / provider.sort — прокидываем их в extra_body
    (OpenRouter: models[] для нативного fallback, provider.sort для выбора
    провайдера). temperature=0.2 — стабильный tool-calling; max_retries=1.
    """
    enforce_price_caps_once(rag_settings)
    extra_body = build_routing_fields(rag_settings.llm_model, rag_settings)
    kwargs = {"extra_body": extra_body} if extra_body else {}
    return ChatOpenAI(
        model=rag_settings.llm_model,
        base_url=rag_settings.llm_base_url,
        api_key=rag_settings.llm_api_key,
        temperature=0.2,
        max_retries=1,
        **kwargs,
    )
