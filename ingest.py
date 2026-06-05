"""
ingest.py — индексирует все документы из папки docs/ в Postgres.

Что делает:
1. Находит все .md и .txt файлы в указанной папке.
2. Для каждого файла:
    a) читает текст;
    b) режет на чанки;
    c) удаляет старые чанки этого файла из БД (на случай переиндексации);
    d) считает эмбеддинги пачками;
    e) кладёт в таблицу chunks.

Запуск:
    python ingest.py                 # индексирует docs/
    python ingest.py path/to/folder  # другую папку
    python ingest.py --batch-size 32 # размер батча для эмбеддингов
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.chunker import chunk_text
from rag.config import settings
from rag.embedder import Embedder, make_embedder
from rag.vector_store import ChunkInput, VectorStore


# Какие расширения считаем «документами».
SUPPORTED_EXTS = {".md", ".txt"}


def find_documents(folder: Path) -> list[Path]:
    """Рекурсивно собирает все поддерживаемые файлы из папки."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Папка не найдена: {folder}")
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    return files


def batched(items: list, size: int):
    """Делит список на батчи фиксированного размера (последний может быть короче)."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def ingest_file(
    path: Path,
    embedder: Embedder,
    store: VectorStore,
    batch_size: int,
) -> int:
    """
    Индексирует один файл. Возвращает количество сохранённых чанков.

    Использует относительное имя файла как `source` — короче и переносимо
    между машинами. Например, "python_basics.md" вместо полного пути.
    """
    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(
        text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    if not chunks:
        print(f"  ⚠ Пустой документ, пропускаю: {path.name}")
        return 0

    source = path.name  # "python_basics.md"

    # Сначала удаляем старые чанки этого источника — это делает индексацию
    # идемпотентной: запустил скрипт второй раз, ничего не задвоилось.
    deleted = store.delete_source(source)
    if deleted:
        print(f"  • удалил {deleted} старых чанков для {source}")

    total = 0
    # Эмбеддим пачками: меньше HTTP-вызовов = быстрее.
    # ВНИМАНИЕ: Voyage принимает до 1000 текстов за запрос.
    # При ошибках лимита уменьшите --batch-size.
    for batch in batched(chunks, batch_size):
        texts = [c.text for c in batch]
        vectors = embedder.embed_documents(texts)
        records = [
            ChunkInput(
                source=source,
                chunk_index=chunk.index,
                content=chunk.text,
                embedding=vector,
            )
            for chunk, vector in zip(batch, vectors)
        ]
        store.insert_chunks(records)
        total += len(records)
        print(f"  • {source}: проиндексировано {total}/{len(chunks)} чанков")

    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Индексация документов в pgvector.")
    parser.add_argument(
        "folder",
        nargs="?",
        default="docs",
        help="Папка с документами (.md/.txt). По умолчанию: docs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Сколько чанков отправлять в Voyage за один запрос. По умолчанию: 16",
    )
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    print(f"Сканирую {folder} …")
    files = find_documents(folder)
    if not files:
        print(
            f"В {folder} не найдено файлов с расширениями {SUPPORTED_EXTS}. "
            f"Положите туда хотя бы один .md и запустите снова."
        )
        return 1
    print(f"Найдено файлов: {len(files)}")

    # Открываем embedder и store через context manager, чтобы гарантированно
    # закрыть HTTP-клиент и соединение с БД даже при ошибке.
    with make_embedder() as embedder, VectorStore() as store:
        total = 0
        for path in files:
            print(f"\n→ {path.name}")
            total += ingest_file(path, embedder, store, args.batch_size)
        print(f"\n✓ Готово. Всего чанков в БД сейчас: {store.count()}")
        print(f"  (за этот запуск проиндексировано: {total})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
