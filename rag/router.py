"""
router.py — классификатор запросов: нужен ли RAG для ответа?

В продакшен-системах «RAG-as-default» — антипаттерн: на «Привет», «Спасибо»
и мета-вопросы про сам диалог не нужен поиск в базе. Это тратит секунды и
выдаёт галлюцинации.

Архитектурное решение — **router**: маленький LLM-классификатор перед
retrieve-этапом. Если запрос не требует knowledge lookup'а, пайплайн
сворачивается до прямого ответа LLM (с историей чата или без).

Четыре категории:
  - "knowledge" — фактический вопрос про предметную область,
                  на который ответ может быть в документах.
                  Пример: «Что такое GIL?», «Чем отличается SQL и NoSQL»
  - "chitchat"  — приветствия, простые реплики, social niceties.
                  Пример: «Привет», «Спасибо», «Как дела»
  - "meta"      — вопрос про сам диалог или предыдущие сообщения.
                  Пример: «Что я спрашивал?», «Повтори последний ответ»
                  Резолвится с историей чата, БЕЗ RAG.
  - "other"     — out-of-scope (погода, новости, личные вопросы).
                  Пример: «Какая завтра погода?», «Кто президент»
                  Честно отказываем.

Поведение fallback: если LLM вернула мусор — считаем "knowledge" и идём
обычным путём (безопаснее лишний раз сделать retrieve, чем пропустить
действительно нужный поиск).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rag.generator import ChatGenerator


# Допустимые intent'ы. Если LLM вернёт что-то вне этого набора — fallback.
VALID_INTENTS = {"knowledge", "chitchat", "meta", "other"}


@dataclass(frozen=True)
class RouteDecision:
    """Результат классификации."""

    intent: str               # из VALID_INTENTS
    reason: str               # короткое обоснование от LLM
    raw_response: str = ""    # сырой ответ для отладки
    fallback: bool = False    # сработал ли fallback на "knowledge"


ROUTER_SYSTEM = """\
Ты — классификатор запросов в RAG-системе. Твоя задача — решить, нужен ли поиск в базе знаний для ответа.

Категории:
  • "knowledge" — фактический вопрос про термины, концепции, технические темы. Ответ может быть в документах базы.
        Примеры: "Что такое GIL?", "Чем отличается SQL и NoSQL?", "Расскажи про индексы"
  • "chitchat"  — приветствия, благодарности, простые реплики. RAG не нужен.
        Примеры: "Привет", "Спасибо", "Доброе утро", "Как дела"
  • "meta"      — вопросы про сам диалог: что говорил пользователь раньше, повтори ответ, объясни проще.
        Резолвится через историю чата, БЕЗ retrieve.
        Примеры: "Что я спрашивал?", "Объясни проще", "Повтори"
  • "other"     — запросы вне темы базы знаний: погода, новости, личные вопросы, политика.
        Примеры: "Какая погода?", "Кто президент", "Что приготовить на ужин"

Правила:
1. Если есть сомнения между "knowledge" и чем-то ещё — выбирай "knowledge" (RAG безопаснее запустить лишний раз).
2. Не путай "meta" с "knowledge": «расскажи подробнее про GIL» это knowledge (раскрытие темы), а «повтори что говорил выше» это meta (про сам диалог).
3. История даётся для контекста — НЕ классифицируй её сообщения, классифицируй только текущий запрос.

Верни строго JSON: {"intent": "...", "reason": "коротко по-русски"}.
Никаких пояснений и markdown.

Примеры:
вопрос: "Привет"
ответ: {"intent": "chitchat", "reason": "приветствие, нет фактического вопроса"}

вопрос: "Что такое ACID?"
ответ: {"intent": "knowledge", "reason": "технический вопрос про базы данных"}

вопрос: "Что я спрашивал в прошлый раз?"
ответ: {"intent": "meta", "reason": "вопрос про предыдущие сообщения диалога"}

вопрос: "Какая погода в Москве?"
ответ: {"intent": "other", "reason": "погода — вне темы базы знаний"}
"""


# Регулярка для извлечения JSON-объекта даже из ответа с лишним текстом.
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


class QueryRouter:
    """
    Делегирует LLM работу по классификации.
    Использует тот же chat-endpoint что и основной генератор.
    """

    def __init__(self, generator: ChatGenerator) -> None:
        self._generator = generator

    def classify(
        self,
        query: str,
        history: list | None = None,
    ) -> RouteDecision:
        """
        Классифицирует текущий запрос. history (если передан) — список
        Message-объектов, добавляется в user-prompt для контекста
        (важно для meta-вопросов).
        """
        if not query or not query.strip():
            return RouteDecision(
                intent="knowledge", reason="пустой запрос", fallback=True,
            )

        # Собираем user-message: краткая история (если есть) + текущий запрос.
        user_parts: list[str] = []
        if history:
            # Берём только последние 4 сообщения — больше для классификации
            # не нужно и удлинит prompt без пользы.
            tail = history[-4:]
            hist_block = "\n".join(
                # Усечём длинные сообщения, чтобы prompt не раздувался.
                f"{m.role}: {m.content if len(m.content) < 200 else m.content[:200] + '…'}"
                for m in tail
            )
            user_parts.append(f"История последних сообщений:\n{hist_block}")
        user_parts.append(f"Текущий запрос: {query}")
        user_msg = "\n\n".join(user_parts)

        payload = {
            "model": self._generator._model,
            "stream": False,
            # Детерминированно — нам нужна стабильная классификация.
            "temperature": 0.0,
            # Reasoning-модель + JSON-ответ — даём бюджет.
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        }
        url = f"{self._generator._base_url}/chat/completions"
        response = self._generator._client.post(url, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"] or ""

        return self._parse(content)

    @staticmethod
    def _parse(content: str) -> RouteDecision:
        """
        Парсит JSON-объект. Терпит markdown-обёртку и лишний текст.
        При любых проблемах — fallback на "knowledge" (безопаснее запустить
        RAG, чем пропустить нужный поиск).
        """
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]

        match = _JSON_OBJECT_RE.search(text)
        if not match:
            return RouteDecision(
                intent="knowledge",
                reason="router не дал JSON, fallback",
                raw_response=content,
                fallback=True,
            )

        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return RouteDecision(
                intent="knowledge",
                reason="router вернул невалидный JSON, fallback",
                raw_response=content,
                fallback=True,
            )

        intent = obj.get("intent")
        reason = obj.get("reason") or ""
        if not isinstance(intent, str) or intent not in VALID_INTENTS:
            return RouteDecision(
                intent="knowledge",
                reason=f"неизвестный intent «{intent}», fallback",
                raw_response=content,
                fallback=True,
            )

        return RouteDecision(
            intent=intent,
            reason=str(reason)[:300],
            raw_response=content,
        )
