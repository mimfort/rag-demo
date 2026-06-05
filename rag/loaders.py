"""
loaders.py — парсеры пользовательских файлов в простой текст.

Поддерживаемые форматы:
  - .md, .markdown, .txt — читаем как UTF-8 (с фоллбэком на cp1251)
  - .pdf                  — pypdf, конкатенация текста со всех страниц

Не сохраняем файлы на диск — работаем с байтами в памяти. После парсинга
получаем текст и забываем про файл; в БД летят только чанки и эмбеддинги.
Это упрощает деплой и снимает вопросы безопасности с хранением загрузок.

Если расширение не поддерживается или файл не парсится — поднимаем
UnsupportedFile с понятным сообщением (его потом превратим в HTTP 400).
"""

from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader


# Лимит: 5 МБ. На нашем учебном демо больше — не нужно, и больше — это
# уже дольше эмбеддить (больше запросов к Voyage).
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

# Расширения которые мы умеем парсить. Сравнение по нижнему регистру.
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}


class UnsupportedFile(Exception):
    """Файл не поддерживается или повреждён. Серверу — превратить в HTTP 400."""


def load_text(filename: str, data: bytes) -> str:
    """
    Возвращает текстовое содержимое файла.

    filename — оригинальное имя (для определения расширения).
    data     — сырые байты файла (max MAX_UPLOAD_BYTES).
    """
    if not filename or not data:
        raise UnsupportedFile("Пустой файл")

    if len(data) > MAX_UPLOAD_BYTES:
        raise UnsupportedFile(
            f"Файл слишком большой: {len(data)} байт, лимит {MAX_UPLOAD_BYTES}"
        )

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFile(
            f"Расширение {ext or '(без расширения)'} не поддерживается. "
            f"Доступно: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        return _load_pdf(data)
    return _load_text_file(data)


def _load_text_file(data: bytes) -> str:
    """
    Текстовый файл. Пробуем UTF-8, при ошибке — cp1251 (старые виндовые .txt).
    Если и это не сработало — поднимаем UnsupportedFile.
    """
    for encoding in ("utf-8", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnsupportedFile("Не удалось распознать кодировку текстового файла")


def _load_pdf(data: bytes) -> str:
    """
    PDF через pypdf. extract_text() работает построчно — для большинства
    «обычных» PDF (отчёты, документация) достаёт текст вполне. Для
    отсканированных PDF (картинок) вернёт пустоту — для OCR нужен
    отдельный инструмент (tesseract), здесь не подключаем.

    Текст со страниц склеиваем двумя \n — это естественные границы для
    smart chunker'а (он режет по абзацам).
    """
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # pypdf может бросить разные виды ошибок
        raise UnsupportedFile(f"PDF не открывается: {exc}") from exc

    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            parts.append(text)

    full = "\n\n".join(parts)
    if not full.strip():
        raise UnsupportedFile(
            "Не удалось извлечь текст из PDF — возможно это отсканированный "
            "документ (картинка). Для OCR нужен отдельный инструмент."
        )
    return full
