"""
generator.py — шаг «Generation» в RAG.

Берёт найденные чанки + вопрос пользователя, формирует промпт и шлёт в
chat-модель LM Studio. Поддерживает два режима:
  - generate()        — обычный ответ строкой
  - generate_stream() — генератор, который выдаёт куски ответа по мере поступления

API чата в LM Studio тоже OpenAI-совместимый:

    POST {base}/chat/completions
    {
      "model": "...",
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user",   "content": "..."}
      ],
      "stream": false | true,
      "temperature": 0.2
    }

В режиме stream=true сервер отвечает Server-Sent Events: серия строк
вида `data: {...json...}\n\n` и в конце `data: [DONE]`.
"""

from __future__ import annotations

import json
from typing import Iterable, Iterator

import httpx

from rag.config import settings
from rag.vector_store import RetrievedChunk


# Системный промпт задаёт «характер» модели и правила игры.
# Важные элементы:
#   - использовать ТОЛЬКО переданный контекст;
#   - честно говорить «не знаю», если контекста не хватает (борьба с галлюцинациями);
#   - вставлять ссылки на конкретные фрагменты вида [1], [2] — на фронте мы
#     будем эти маркеры парсить и делать кликабельными, чтобы пользователь
#     видел источник каждого утверждения.
SYSTEM_PROMPT = (
    "Ты — ассистент, отвечающий на вопросы пользователя СТРОГО по предоставленному "
    "контексту. Контекст состоит из пронумерованных фрагментов вида «[Фрагмент N]».\n\n"
    "Правила:\n"
    "1. Каждое утверждение в ответе подкрепляй ссылкой на номер фрагмента "
    "в квадратных скобках: [1], [2]. Если утверждение подкреплено несколькими "
    "фрагментами, ставь подряд: [1][3]. Без таких ссылок ответ не принимается.\n"
    "2. Если в контексте нет ответа на вопрос — прямо скажи «В контексте нет "
    "информации по этому вопросу» и не выдумывай.\n"
    "3. Не пересказывай контекст целиком. Отвечай конкретно на вопрос, "
    "коротко и по делу. Списки и форматирование — приветствуются.\n"
    "4. Не дублируй раздел «Источники» в конце — номера [N] уже играют эту роль."
)


def build_user_prompt(query: str, chunks: Iterable[RetrievedChunk]) -> str:
    """
    Склеивает контекст и вопрос в user-сообщение.

    Формат блока контекста — пронумерованные фрагменты. Номер N в заголовке
    «[Фрагмент N]» совпадает с позицией чанка в top-K на фронте, поэтому
    ссылки [N] в ответе LLM будут идеально матчиться с карточками в UI.
    """
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[Фрагмент {i} | источник: {chunk.source}]\n"
            f"{chunk.content}"
        )
    context = "\n\n".join(blocks) if blocks else "(контекст пуст)"

    return (
        "Используй приведённый ниже контекст для ответа на вопрос. "
        "В ответе обязательно ставь ссылки [N] на номера фрагментов.\n\n"
        f"=== КОНТЕКСТ ===\n{context}\n=== КОНЕЦ КОНТЕКСТА ===\n\n"
        f"Вопрос: {query}"
    )


class LMStudioGenerator:
    """
    Клиент к /v1/chat/completions LM Studio.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = (base_url or settings.lm_studio_base_url).rstrip("/")
        self._api_key = api_key or settings.lm_studio_api_key
        self._model = model or settings.chat_model
        self._timeout = timeout
        # Для streaming делаем отдельный клиент с большим таймаутом —
        # streaming-ответ может «капать» минутами.
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    # --- обычный (нестриминговый) ответ ---

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        temperature: float = 0.2,
    ) -> str:
        """
        Шлёт запрос и ждёт весь ответ целиком. Возвращает текст ответа.
        temperature=0.2 — почти детерминированно (0 — совсем; 1+ — креативно).
        Для RAG обычно ставят низкую температуру: нам нужен точный ответ
        по контексту, а не «художественный пересказ».
        """
        payload = self._build_payload(query, chunks, temperature, stream=False)
        url = f"{self._base_url}/chat/completions"
        response = self._client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        # OpenAI-совместимый ответ: {"choices": [{"message": {"content": "..."}}]}
        return data["choices"][0]["message"]["content"]

    # --- стриминговый ответ ---

    def generate_stream(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """
        Генератор: возвращает «кусочки» ответа по мере того, как модель
        их выдаёт. Это даёт классический эффект «печатной машинки».

        Под капотом — Server-Sent Events. Сервер шлёт строки вида:
            data: {"choices":[{"delta":{"content":"Прив"}}]}
            data: {"choices":[{"delta":{"content":"ет"}}]}
            ...
            data: [DONE]

        Мы парсим JSON каждой строки и выдаём delta.content наружу.
        """
        payload = self._build_payload(query, chunks, temperature, stream=True)
        url = f"{self._base_url}/chat/completions"

        # stream=True у httpx означает «не загружай ответ в память целиком,
        # дай мне читать построчно по мере поступления».
        with self._client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                # SSE-формат: каждая значимая строка начинается с "data: "
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):].strip()
                # Маркер конца стрима.
                if data_str == "[DONE]":
                    break
                # Иногда сервер шлёт пустые keep-alive строки — пропустим то,
                # что не парсится как JSON.
                try:
                    payload_chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                # Из delta нас интересует только content: новые символы текста.
                # В первом чанке может прилететь только role=assistant без content.
                choices = payload_chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece

    # --- общая часть ---

    def _build_payload(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        temperature: float,
        stream: bool,
    ) -> dict:
        return {
            "model": self._model,
            "stream": stream,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(query, chunks)},
            ],
        }

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LMStudioGenerator":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
