"""
Фабрика LLM. ChatOpenAI работает с любым OpenAI-совместимым провайдером
(по умолчанию OpenRouter): отличается только base_url/ключ/модель в .env.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from rag.config import settings as rag_settings


def make_llm() -> ChatOpenAI:
    """
    Возвращает ChatOpenAI, настроенную на chat-провайдера из .env
    (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL).

    temperature=0.2 — нам нужен стабильный tool-calling, не креатив.
    max_retries=1 — короткий retry; при недоступности провайдера лучше
    упасть явно, чем долго ретраить.
    """
    return ChatOpenAI(
        model=rag_settings.llm_model,
        base_url=rag_settings.llm_base_url,
        api_key=rag_settings.llm_api_key,
        temperature=0.2,
        max_retries=1,
    )
