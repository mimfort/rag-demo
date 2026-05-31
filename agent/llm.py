"""
Фабрика LLM. ChatOpenAI работает с LM Studio: тот OpenAI-совместим,
просто base_url другой.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from rag.config import settings as rag_settings


def make_llm() -> ChatOpenAI:
    """
    Возвращает ChatOpenAI настроенную на локальную LM Studio.

    temperature=0.2 — мы хотим стабильный tool-calling, не креатив.
    max_retries=1 — обычно LM Studio либо отвечает, либо нет; retry не
    спасёт.
    """
    return ChatOpenAI(
        model=rag_settings.chat_model,
        base_url=rag_settings.lm_studio_base_url,
        api_key=rag_settings.lm_studio_api_key,
        temperature=0.2,
        max_retries=1,
    )
