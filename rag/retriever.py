"""
retriever.py — оркестрирует шаг «Retrieval» в RAG.

Что делает этот модуль:
  пользователь дал текст вопроса
  → мы эмбеддим вопрос через ту же модель, что и документы
  → ищем top_k ближайших чанков в pgvector
  → возвращаем их вместе со score'ами.

Почему важно использовать ТУ ЖЕ модель для запроса и для документов:
эмбеддинг-пространства разных моделей несовместимы. У bge-m3 свой
«внутренний язык», у text-embedding-3-small — другой. Если документы
индексированы одной моделью, а запрос эмбеддим другой — получим мусор.
"""

from __future__ import annotations

from rag.embedder import Embedder
from rag.vector_store import RetrievedChunk, VectorStore


class Retriever:
    """
    Связывает Embedder и VectorStore. Сам ничего не хранит — это просто склейка.
    """

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """
        Главный метод: текст вопроса → список релевантных чанков.

        Можно сюда дописать query rewriting (переформулировать вопрос
        перед эмбеддингом), HyDE (сгенерить «гипотетический ответ» и
        эмбеддить уже его), reranker (отдельной моделью переупорядочить
        top_k). Это типичные улучшения, но базовый RAG обходится без них.
        """
        query_vec = self._embedder.embed_query(query)
        return self._store.search(query_vec, top_k=top_k)
