"""
history.py — управление историей разговора и переформулировка
context-зависимых запросов.

Зачем:
  «А расскажи подробнее про X» — без контекста LLM-retriever не знает что
  такое X. На уровне эмбеддинга «расскажи подробнее» матчится почти со
  всем — поиск выдаёт мусор.

  Решение классическое: перед retrieve берём последние N сообщений чата
  и просим LLM **переформулировать текущий вопрос в STANDALONE вид** с
  явными существительными вместо местоимений и общих фраз.

Пример:
  history: [user: «Что такое GIL?», assistant: «GIL — это мьютекс ...»]
  cur:     «А расскажи подробнее»
       ↓ LLM standalone-rewriter
  standalone: «Расскажи подробнее про GIL в Python»

Хранение:
  - Таблица messages в Postgres. ChatHistoryStore — обёртка с двумя
    методами: save_message, get_recent.
  - Связь с chats через FK + ON DELETE CASCADE (удалил чат — стерлись
    его сообщения).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import psycopg

from rag.generator import LMStudioGenerator


@dataclass(frozen=True)
class Message:
    role: str       # "user" | "assistant"
    content: str
    created_at: str | None = None  # ISO timestamp, для UI
    # Опциональные поля для assistant-сообщений — сохраняются в БД
    # чтобы можно было показать «как я дошёл до ответа» после рестарта.
    chunks: list | None = None    # список dict'ов (RetrievedChunk JSON)
    explain: dict | None = None
    prompt: str | None = None


class ChatHistoryStore:
    """
    Тонкая обёртка над таблицей messages.
    Использует то же psycopg-соединение что и VectorStore — соединение
    передаётся снаружи (не плодим лишних коннектов).
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def save_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        *,
        chunks: list | None = None,
        explain: dict | None = None,
        prompt: str | None = None,
    ) -> None:
        """
        Записывает одно сообщение. Для assistant-сообщений можно (и нужно)
        сохранять полный pipeline-снимок: chunks/explain/prompt — чтобы
        после рестарта показать «как я дошёл до ответа».
        psycopg сам сериализует list/dict в jsonb через адаптер Json().
        """
        import json
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages (chat_id, role, content, chunks, explain, prompt)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    chat_id,
                    role,
                    content,
                    json.dumps(chunks) if chunks is not None else None,
                    json.dumps(explain) if explain is not None else None,
                    prompt,
                ),
            )

    def get_recent(
        self, chat_id: str, limit: int = 6, with_details: bool = False,
    ) -> list[Message]:
        """
        Возвращает последние N сообщений чата в хронологическом порядке.

        with_details=False — только role/content/created_at (для standalone-
        rewrite это всё что нужно, не таскаем гигантские explain-объекты).
        with_details=True  — плюс chunks/explain/prompt для UI.
        """
        cols = "role, content, created_at"
        if with_details:
            cols += ", chunks, explain, prompt"

        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {cols}
                FROM messages
                WHERE chat_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (chat_id, limit),
            )
            rows = cur.fetchall()
        # Перевернём в хронологический порядок (старые в начале).
        rows.reverse()

        result: list[Message] = []
        for r in rows:
            kwargs = {
                "role": r[0],
                "content": r[1],
                "created_at": r[2].isoformat() if r[2] is not None else None,
            }
            if with_details:
                # psycopg возвращает jsonb уже распарсенным в Python-объекты.
                kwargs["chunks"] = r[3]
                kwargs["explain"] = r[4]
                kwargs["prompt"] = r[5]
            result.append(Message(**kwargs))
        return result

    def clear(self, chat_id: str) -> int:
        """Удаляет все сообщения чата (используется при удалении чата)."""
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM messages WHERE chat_id = %s", (chat_id,))
            return cur.rowcount


# ---------------------------------------------------------------------------
# Standalone rewriter — превращает context-зависимый запрос в самодостаточный
# ---------------------------------------------------------------------------

_STANDALONE_SYSTEM = """\
Ты — переформулятор поисковых запросов в диалоге. Получаешь:
  • историю последних сообщений между пользователем и ассистентом;
  • текущий запрос пользователя.

Твоя задача: переписать ТЕКУЩИЙ запрос в самодостаточный (standalone) вид так, чтобы его можно было понять без истории.

Правила:
1. Заменяй местоимения и сокращения («он», «это», «там», «подробнее») на конкретные сущности из истории.
2. Если текущий запрос УЖЕ самодостаточный — верни его БЕЗ изменений.
3. Не выдумывай новые сущности. Используй только то что упоминалось в истории или текущем запросе.
4. Сохраняй язык запроса (русский → русский).
5. Краткость важнее: не раздувай переформулировку лишними фактами из истории.

Верни JSON: {"standalone": "переформулированный запрос"}.
Никаких пояснений и markdown.

Примеры:

История:
user: Что такое GIL?
assistant: GIL — это мьютекс в CPython, который позволяет только одному потоку...
Текущий запрос: А расскажи подробнее
Ответ: {"standalone": "Расскажи подробнее про GIL в Python"}

История:
user: В чём разница SQL и NoSQL?
assistant: SQL — реляционные базы со строгой схемой...
Текущий запрос: А как с этим у MongoDB?
Ответ: {"standalone": "Как с моделью данных и схемой у MongoDB и какие компромиссы?"}

История:
user: Что такое HNSW?
assistant: HNSW — это approximate nearest neighbors алгоритм...
Текущий запрос: Что такое ACID?
Ответ: {"standalone": "Что такое ACID?"}
"""


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


@dataclass(frozen=True)
class StandaloneResult:
    """Что вышло после переформулировки."""

    standalone: str       # переформулированный (или исходный) запрос
    changed: bool         # отличается ли от исходного
    raw_response: str = ""  # сырой ответ LLM для отладки


def make_standalone_query(
    generator: LMStudioGenerator,
    history: list[Message],
    current_query: str,
) -> StandaloneResult:
    """
    Переформулирует current_query в standalone-вид с учётом истории.
    Если history пуст — возвращает current_query без изменений (там
    нечего разрешать).
    """
    if not history:
        return StandaloneResult(standalone=current_query, changed=False)

    # Форматируем историю в краткий, читаемый для LLM формат.
    lines = []
    for m in history:
        # Усечём слишком длинные ответы ассистента — для разрешения
        # местоимений хватает первой пары предложений, не нужен весь текст.
        content = m.content if len(m.content) < 400 else m.content[:400] + "…"
        lines.append(f"{m.role}: {content}")
    history_block = "\n".join(lines)

    user_msg = (
        f"История:\n{history_block}\n\n"
        f"Текущий запрос: {current_query}"
    )

    payload = {
        "model": generator._model,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 1024,  # с запасом на reasoning + JSON-ответ
        "messages": [
            {"role": "system", "content": _STANDALONE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    }
    url = f"{generator._base_url}/chat/completions"
    response = generator._client.post(url, json=payload)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"] or ""

    # Парсим JSON.
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return StandaloneResult(
            standalone=current_query, changed=False, raw_response=content,
        )
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return StandaloneResult(
            standalone=current_query, changed=False, raw_response=content,
        )
    standalone = obj.get("standalone")
    if not isinstance(standalone, str) or not standalone.strip():
        return StandaloneResult(
            standalone=current_query, changed=False, raw_response=content,
        )
    standalone = standalone.strip()
    return StandaloneResult(
        standalone=standalone,
        changed=(standalone != current_query.strip()),
        raw_response=content,
    )
