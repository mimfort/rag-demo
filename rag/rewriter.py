"""
rewriter.py — query rewriting через LLM.

Отличие от decomposer:
  - decomposer  ⇒ «Что такое GIL и ACID?» → ["Что такое GIL?", "Что такое ACID?"]
                 (разделение по ТЕМАМ — атомарные разные подвопросы)
  - rewriter    ⇒ «Что такое GIL?»        → ["Что такое GIL?",
                                              "Объясни Global Interpreter Lock в Python",
                                              "Почему многопоточный Python не ускоряет CPU-bound задачи"]
                 (одна и та же тема, РАЗНЫЕ формулировки)

Зачем нужен rewriter:
  bi-encoder (наш bge-m3) кодирует текст в один вектор. Разные слова про
  одно и то же могут давать ЗАМЕТНО разные векторы — например модель видит
  «асинхронность» и «параллельность» близкими, а «GIL» и «многопоточность»
  далёкими (хотя для человека понятно что связаны).

  Делая несколько эмбеддингов разных формулировок и сливая кандидатов через
  RRF, мы расширяем «зону покрытия» в семантическом пространстве. Чанк,
  который модель не нашла на исходную формулировку, может всплыть на
  переформулировке через технические термины.

Поведение fallback: если LLM вернул мусор/пусто — используем массив из
одного элемента (исходного запроса). Тогда pipeline degrade'ится в обычный
single-query retrieve.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rag.generator import ChatGenerator


@dataclass(frozen=True)
class RewriteResult:
    """
    Результат с явным статусом.

    Статусы:
      - "rewritten" — LLM вернул валидный массив с дополнительными формулировками.
      - "failed"    — мусор, используем fallback на оригинал.
    """

    rewrites: list[str]
    status: str               # "rewritten" | "failed"
    raw_response: str = ""    # сырой ответ LLM — для отладки


# System prompt: чёткие правила и примеры. Просим РАЗНОТИПНЫЕ
# переформулировки: технические термины, синонимы, развёрнутый и
# сжатый варианты. На Gemma 4B-E2B стабильно работает.
REWRITER_SYSTEM = """\
Ты — оптимизатор поисковых запросов. Сгенерируй N разных формулировок одного и того же вопроса пользователя, сохраняя его смысл, но используя:
  • синонимы и парафразы;
  • технические термины (например, «многопоточность» вместо «параллельность»);
  • развёрнутую версию с пояснением контекста;
  • краткую версию с ключевыми словами.

Правила:
1. ИСХОДНЫЙ запрос всегда должен быть первым элементом массива (без изменений).
2. Сохраняй язык исходного запроса (русский → русский).
3. Не выдумывай контекст которого нет в исходном вопросе.
4. Все формулировки должны быть про ТО ЖЕ САМОЕ. Не разделяй запрос на разные темы — для этого есть отдельный декомпозитор.

Верни строго JSON-массив строк длины N. Только JSON, без markdown-блоков и пояснений.

Примеры (N=3):
вопрос: "Что такое GIL?"
ответ: ["Что такое GIL?", "Объясни Global Interpreter Lock в Python", "Почему многопоточный Python не ускоряет CPU-bound задачи"]

вопрос: "Что такое индексы в БД?"
ответ: ["Что такое индексы в БД?", "Зачем нужны индексы в базе данных и какие они бывают", "B-tree GIN GiST hash-индексы"]
"""


# Регулярка чтобы вытащить JSON-массив даже из ответа с лишним текстом.
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


class QueryRewriter:
    """
    Через тот же ChatGenerator (chat-модель) генерирует N переформулировок.
    """

    def __init__(self, generator: ChatGenerator) -> None:
        self._generator = generator

    def rewrite(self, query: str, n: int = 3) -> RewriteResult:
        """
        Возвращает RewriteResult из N формулировок. n должно быть 1..5.
        При n=1 — фактически no-op (всегда возвращает [query]).
        """
        if not query or not query.strip():
            return RewriteResult([], status="failed")
        # n=1 — нет смысла дергать LLM, возвращаем исходный.
        if n <= 1:
            return RewriteResult([query], status="rewritten")

        # В user-сообщение явно пишем сколько вариантов хотим — LLM нужна
        # эта подсказка чтобы не вернуть 2 или 5 вместо запрошенных 3.
        user_msg = f"N = {n}\nвопрос: {query}"

        payload = {
            "model": self._generator._model,
            "stream": False,
            "temperature": 0.3,  # чуть выше нуля — нужно разнообразие формулировок.
            "max_tokens": 1024,   # с запасом на reasoning + N коротких строк.
            "messages": [
                {"role": "system", "content": REWRITER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        }
        url = f"{self._generator._base_url}/chat/completions"
        response = self._generator._client.post(url, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"] or ""

        return self._parse(content, fallback=query, n=n)

    @staticmethod
    def _parse(content: str, fallback: str, n: int) -> RewriteResult:
        """
        Парсит JSON-массив. Допускает обрамление ```json … ``` и
        лишний текст до/после массива. Если ничего не вытащили —
        fallback на массив с одним исходным запросом.
        """
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]

        match = _JSON_ARRAY_RE.search(text)
        if not match:
            return RewriteResult([fallback], status="failed", raw_response=content)

        try:
            arr = json.loads(match.group(0))
        except json.JSONDecodeError:
            return RewriteResult([fallback], status="failed", raw_response=content)

        if not isinstance(arr, list):
            return RewriteResult([fallback], status="failed", raw_response=content)

        cleaned = [s.strip() for s in arr if isinstance(s, str) and s.strip()]
        if not cleaned:
            return RewriteResult([fallback], status="failed", raw_response=content)

        # Если LLM вернула меньше чем просили — это OK, не failure.
        # Если больше N — обрежем (берём первые N).
        if len(cleaned) > n:
            cleaned = cleaned[:n]

        # Гарантия: исходный запрос всегда первый. Если LLM его пропустила
        # или переформулировала — подставим обратно в начало.
        if cleaned[0] != fallback:
            cleaned = [fallback] + cleaned[: n - 1]

        return RewriteResult(cleaned, status="rewritten", raw_response=content)
