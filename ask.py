"""
ask.py — задаёт вопрос RAG-системе.

Что делает:
1. Эмбеддит вопрос пользователя.
2. Ищет top-K похожих чанков в pgvector.
3. Формирует промпт с найденным контекстом.
4. Получает ответ от chat-модели (обычный или streaming).

Запуск:
    python ask.py "Что такое GIL в Python?"
    python ask.py --stream "Объясни ACID"
    python ask.py --verbose "Как работает HNSW?"
    python ask.py -k 3 "Что такое REST?"

Флаг --verbose очень полезен для понимания — он показывает весь pipeline:
какие чанки найдены, с какими score'ами, какой промпт ушёл в LLM.
"""

from __future__ import annotations

import argparse
import sys

from rag.config import settings
from rag.embedder import LMStudioEmbedder
from rag.generator import LMStudioGenerator, build_user_prompt
from rag.retriever import Retriever
from rag.vector_store import RetrievedChunk, VectorStore


def print_retrieved_chunks(chunks: list[RetrievedChunk]) -> None:
    """Печатает найденные чанки в человекочитаемом виде."""
    if not chunks:
        print("  (ничего не найдено — база пуста? Запусти python ingest.py)")
        return
    for i, chunk in enumerate(chunks, start=1):
        # Превью — первые 200 символов чанка, чтобы не загромождать вывод.
        preview = chunk.content[:200].replace("\n", " ")
        if len(chunk.content) > 200:
            preview += "…"
        print(
            f"  [{i}] similarity={chunk.similarity:.4f}  source={chunk.source}\n"
            f"      {preview}"
        )


def run(query: str, top_k: int, stream: bool, verbose: bool) -> int:
    # Открываем все три ресурса разом: эмбеддер, хранилище, генератор.
    with (
        LMStudioEmbedder() as embedder,
        VectorStore() as store,
        LMStudioGenerator() as generator,
    ):
        retriever = Retriever(embedder, store)

        if verbose:
            print(f"\n[1/3] Эмбеддим вопрос моделью {settings.embedding_model}…")

        chunks = retriever.retrieve(query, top_k=top_k)

        if verbose:
            print(f"\n[2/3] Найдено чанков (top-{top_k}):")
            print_retrieved_chunks(chunks)
            print(f"\n[3/3] Формируем промпт и отправляем в {settings.chat_model}…")
            print("─" * 70)
            print("ПРОМПТ (user-сообщение):")
            print("─" * 70)
            print(build_user_prompt(query, chunks))
            print("─" * 70)
            print()

        # Если контекст пуст — нет смысла гонять LLM.
        if not chunks:
            print("Контекст пуст. Сначала проиндексируй документы: python ingest.py")
            return 1

        print("ОТВЕТ:")
        print("─" * 70)
        if stream:
            # Печатаем по мере поступления. flush=True важно: без него Python
            # будет буферизовать вывод и эффект «печатной машинки» пропадёт.
            for piece in generator.generate_stream(query, chunks):
                print(piece, end="", flush=True)
            print()  # перенос строки в конце
        else:
            answer = generator.generate(query, chunks)
            print(answer)
        print("─" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Спросить что-нибудь у RAG.")
    parser.add_argument("query", help="Текст вопроса (в кавычках)")
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=settings.top_k,
        help=f"Сколько чанков подкладывать в контекст. По умолчанию: {settings.top_k}",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Получать ответ по мере генерации (эффект «печатной машинки»).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Показать промежуточные шаги: найденные чанки, итоговый промпт.",
    )
    args = parser.parse_args()

    return run(
        query=args.query,
        top_k=args.top_k,
        stream=args.stream,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
