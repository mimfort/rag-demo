"""
decomposer.py — query decomposition через LLM.

Зачем: bi-encoder (наш эмбеддер) превращает любой текст в **один** вектор
фиксированной длины. Если в запросе несколько разных вопросов, их смыслы
смешиваются — вектор оказывается «посередине» в семантическом пространстве,
и каждый из аспектов получает посредственный score.

Решение — **разложить** составной запрос на атомарные подвопросы перед
retrieve, каждый прогнать отдельно, объединить кандидатов через RRF.

Что делает этот модуль:
  query: «Что такое GIL? Как часто новичок с ним встречается?»
              ↓ LLM с инструкцией декомпозиции
  subqueries: [
      "Что такое GIL?",
      "Как часто новичок сталкивается с GIL?"
  ]

Важная инвариантность: если запрос УЖЕ атомарный — LLM возвращает его
без изменений как массив из одного элемента. Тогда multi-query degenerate'ся
в обычный single-query retrieve.

Fallback: если LLM вернула мусор (не JSON, не массив строк) — используем
original query. Это безопасное поведение, хуже не будет.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rag.generator import ChatGenerator


@dataclass(frozen=True)
class DecomposeResult:
    """
    Результат работы декомпозитора с явным статусом.

    Статусы:
      - "decomposed": LLM вернул валидный массив из 2+ элементов.
      - "atomic":     LLM вернул валидный массив из 1 элемента — запрос
                      признан атомарным.
      - "failed":     LLM вернул мусор (пустая строка, не-JSON) и мы
                      использовали fallback на исходный запрос.

    UI может различать эти случаи: "atomic" — нормальное поведение,
    "failed" — намёк на проблему с моделью или промптом.
    """

    subqueries: list[str]
    status: str               # "decomposed" | "atomic" | "failed"
    raw_response: str = ""    # сырой ответ LLM (для отладки)


# System prompt задаёт характер ответа. Чёткие правила + примеры — у Gemma
# на этом стабильно получается валидный JSON.
DECOMPOSER_SYSTEM = """\
Ты — оптимизатор поисковых запросов. Раздели запрос пользователя на атомарные подвопросы — каждый про ОДНУ тему.

Правила:
1. Если запрос уже атомарный (одна тема, нет «и», «также», нескольких знаков вопроса) — верни его БЕЗ изменений в массиве из одного элемента.
2. Если запрос составной — раздели на 2-4 коротких подвопроса.
3. Каждый подвопрос — самостоятельный, без местоимений ссылающихся на другие подвопросы. Раскрывай «он», «его», «с ним» в явные слова, чтобы каждый подвопрос можно было задать сам по себе.
4. Не добавляй контекст которого нет в исходном запросе.
5. Не переводи на другой язык.

Верни строго JSON-массив строк. Только JSON, без пояснений и без markdown-блоков.

Примеры:
вопрос: "Что такое GIL?"
ответ: ["Что такое GIL?"]

вопрос: "Что такое GIL? Как с ним работает asyncio?"
ответ: ["Что такое GIL?", "Как asyncio работает с GIL?"]

вопрос: "Какая разница между HTTP и WebSocket и зачем нужен Keep-Alive?"
ответ: ["В чём разница между HTTP и WebSocket?", "Зачем нужен Keep-Alive в HTTP?"]
"""


# Чтобы вытащить JSON-массив из ответа модели, даже если она обернёт его
# в ```json ... ``` или добавит лишний текст.
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


class QueryDecomposer:
    """
    Делегирует LLM работу по разбору запроса. Хранит ссылку на готовый
    генератор (тот же, что отвечает на основной вопрос) — не плодит
    лишних HTTP-клиентов.
    """

    def __init__(self, generator: ChatGenerator) -> None:
        self._generator = generator

    def decompose(self, query: str) -> DecomposeResult:
        """
        Возвращает DecomposeResult со статусом и списком подзапросов.
        """
        if not query or not query.strip():
            return DecomposeResult(subqueries=[], status="atomic")

        payload = {
            "model": self._generator._model,
            "stream": False,
            # Детерминированно: для классификации/разбора температуру задирать
            # незачем, нам нужны стабильные одинаковые ответы.
            "temperature": 0.0,
            # 1024 — с запасом на reasoning-токены. Gemma 4 e2b — reasoning-модель,
            # тратит часть бюджета на внутреннее рассуждение перед ответом.
            # На сложных запросах с 256 максимум часть моделей возвращает
            # пустую строку — reasoning «съел» весь бюджет.
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": DECOMPOSER_SYSTEM},
                {"role": "user", "content": query},
            ],
        }
        url = f"{self._generator._base_url}/chat/completions"
        response = self._generator._client.post(url, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"] or ""

        return self._parse(content, fallback=query)

    @staticmethod
    def _parse(content: str, fallback: str) -> DecomposeResult:
        """
        Парсит JSON-массив. Прощает мелкие огрехи модели:
          - оборачивание в markdown-кавычки ```json … ```
          - наличие пояснительного текста до/после массива.
        Если ничего не вытащили — status="failed" + fallback на исходный.
        """
        text = content.strip()

        # Уберём markdown code fence, если есть.
        if text.startswith("```"):
            # пропускаем первую строку (```… или ```json) и последний ```
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]

        # Найдём первый JSON-массив в тексте — на случай если модель
        # написала что-то вроде «Ответ: [...]».
        match = _JSON_ARRAY_RE.search(text)
        if not match:
            return DecomposeResult([fallback], status="failed", raw_response=content)

        try:
            arr = json.loads(match.group(0))
        except json.JSONDecodeError:
            return DecomposeResult([fallback], status="failed", raw_response=content)

        if not isinstance(arr, list):
            return DecomposeResult([fallback], status="failed", raw_response=content)
        cleaned = [s.strip() for s in arr if isinstance(s, str) and s.strip()]
        if not cleaned:
            return DecomposeResult([fallback], status="failed", raw_response=content)
        status = "decomposed" if len(cleaned) > 1 else "atomic"
        return DecomposeResult(cleaned, status=status, raw_response=content)
