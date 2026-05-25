"""
chunker.py — «умный» нарезатель текста на чанки.

Главная идея: текст содержит **естественные границы** (конец абзаца, конец
предложения), и резать лучше по ним, а не по случайным позициям. Тогда
эмбеддинг чанка будет про что-то цельное, а не про обрывок мысли.

Стратегия — recursive splitting:
  1. Если кусок укладывается в chunk_size — отдаём как чанк.
  2. Иначе пробуем разбить его по абзацам (`\n\n`).
  3. Если какой-то «абзац» всё равно длиннее chunk_size — режем по
     предложениям (`. `, `! `, `? `).
  4. Если предложение запредельно длинное (нет нормальных границ) —
     fallback на резку по символам с overlap'ом.

Дополнительно: после получения «атомарных» кусочков мы их **жадно склеиваем**
обратно в чанки до chunk_size, чтобы не плодить много мелких чанков —
маленький контекст бесполезен для LLM.

Overlap делаем не «по символам», а **по последнему предложению** предыдущего
чанка — так стык остаётся читаемым и осмысленным.

Сравнение со старой версией (sliding window по символам):
  - Старый чанк мог начинаться с середины слова: «...едней транзакции и...»
  - Новый чанк всегда начинается с границы предложения/абзаца.
  - Эмбеддинг становится «чище» — модель не путается на обрывках.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """
    Один кусок текста с метаданными.
    `index` — порядковый номер чанка в документе, понадобится при сохранении в БД.
    """

    index: int
    text: str


# Регулярка для разбиения по предложениям.
# Ищем конец предложения: точка/восклицательный/вопросительный, после которого
# идёт пробел и заглавная буква (любого алфавита).
# Знак переноса строки тоже считаем разделителем.
#
# Ограничения: не идеально работает на сокращениях («т.е.», «и т.д.»),
# но для нашего корпуса достаточно. Для production используют NLTK/spaCy/pysbd.
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ])",
    flags=re.UNICODE,
)


def _split_paragraphs(text: str) -> list[str]:
    """
    Делит текст на абзацы.
    Абзацы разделены одной или несколькими пустыми строками (`\n\n+`).
    Лишние пробелы внутри абзаца схлопываем.
    """
    raw = re.split(r"\n\s*\n+", text)
    paragraphs: list[str] = []
    for p in raw:
        clean = " ".join(p.split())  # схлопываем пробелы внутри абзаца
        if clean:
            paragraphs.append(clean)
    return paragraphs


def _split_sentences(paragraph: str) -> list[str]:
    """
    Делит абзац на предложения по регулярке выше.
    Если предложений не нашлось — возвращаем сам абзац как одно «предложение»
    (например, заголовок без точки в конце).
    """
    sentences = _SENTENCE_BOUNDARY.split(paragraph)
    return [s.strip() for s in sentences if s.strip()]


def _split_chars(text: str, max_size: int) -> list[str]:
    """
    Fallback: режем длинный кусок по символам, стараясь рвать
    хотя бы на пробеле.

    Используется только когда у нас «предложение» оказалось чудовищно
    длинным (например, кусок кода или таблица без точек).
    """
    pieces: list[str] = []
    while text:
        if len(text) <= max_size:
            pieces.append(text)
            break
        # Ищем последний пробел до max_size — чтобы не резать слово.
        cut = text.rfind(" ", 0, max_size)
        if cut <= 0:
            cut = max_size  # совсем нет пробелов — режем как есть
        pieces.append(text[:cut].strip())
        text = text[cut:].lstrip()
    return [p for p in pieces if p]


def _atomize(text: str, chunk_size: int) -> list[str]:
    """
    Превращает текст в «атомарные» куски: каждый ≤ chunk_size и каждый —
    цельное предложение или абзац (а не обрывок).

    Это полуфабрикат: дальше мы их жадно склеим в чанки нужного размера.
    """
    atoms: list[str] = []
    for para in _split_paragraphs(text):
        if len(para) <= chunk_size:
            atoms.append(para)
            continue
        # Абзац слишком длинный — режем по предложениям.
        for sent in _split_sentences(para):
            if len(sent) <= chunk_size:
                atoms.append(sent)
            else:
                # Предложение тоже слишком длинное — fallback на символы.
                atoms.extend(_split_chars(sent, chunk_size))
    return atoms


def _build_chunks(
    atoms: list[str], chunk_size: int, overlap_sentences: int
) -> list[Chunk]:
    """
    Жадно склеиваем атомы в чанки до chunk_size.

    Overlap делаем «по последним N атомам» предыдущего чанка — обычно
    это 1-2 последних предложения. Они станут первыми атомами следующего
    чанка, чтобы фраза «на стыке» не потерялась при поиске.
    """
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_size = 0
    index = 0

    def flush() -> None:
        nonlocal buffer, buffer_size, index
        if not buffer:
            return
        text = " ".join(buffer).strip()
        chunks.append(Chunk(index=index, text=text))
        index += 1
        # Готовим следующий буфер из «хвоста» предыдущего — это overlap.
        tail = buffer[-overlap_sentences:] if overlap_sentences > 0 else []
        buffer = list(tail)
        buffer_size = sum(len(a) + 1 for a in buffer)

    for atom in atoms:
        # +1 учитывает пробел-разделитель при склейке.
        prospect = buffer_size + len(atom) + (1 if buffer else 0)
        if prospect > chunk_size and buffer:
            flush()
            # После flush в буфере уже лежит overlap — атом добавим ниже.
            prospect = buffer_size + len(atom) + (1 if buffer else 0)
        buffer.append(atom)
        buffer_size = prospect

    flush()
    return chunks


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """
    Точка входа. Параметры:

      chunk_size — желаемый максимальный размер чанка в символах.
                   Может быть слегка превышен на одно предложение,
                   если оно само длинное.

      overlap — поле принимается ради совместимости со старым API
                (вызовами из ingest.py). В новой реализации overlap
                управляется параметром overlap_sentences.

      overlap_sentences — сколько последних предложений из предыдущего
                          чанка переносить в начало следующего.
                          1 обычно хватает.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным")
    if not text or not text.strip():
        return []

    atoms = _atomize(text, chunk_size)
    return _build_chunks(atoms, chunk_size, overlap_sentences)
