"""
config.py — единая точка чтения настроек.

Все остальные модули импортируют объект `settings` отсюда и не лезут
в os.environ напрямую. Это позволяет в одном месте увидеть весь список
параметров приложения и легко переопределить их в тестах.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Загружаем переменные из файла .env в os.environ.
# Если файла нет — ничего страшного, dotenv просто молча пропустит.
# В этом случае значения возьмутся либо из реального окружения,
# либо из дефолтов ниже.
load_dotenv()


def _env(name: str, default: str | None = None) -> str:
    """
    Достаём строковое значение переменной окружения.
    Если переменная не задана и нет дефолта — падаем с понятным сообщением
    (лучше упасть сразу при старте, чем получить странную ошибку посреди работы).
    """
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(
            f"Переменная окружения {name} не задана. "
            f"Скопируй .env.example в .env и проверь значения."
        )
    return value


def _env_int(name: str, default: int) -> int:
    """То же самое, но конвертирует в int."""
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


@dataclass(frozen=True)
class Settings:
    """
    Иммутабельный (frozen=True) контейнер всех настроек приложения.
    frozen защищает от случайной перезаписи поля в рантайме.
    """

    # --- LM Studio ---
    lm_studio_base_url: str
    lm_studio_api_key: str
    embedding_model: str
    embedding_dim: int
    chat_model: str

    # --- PostgreSQL ---
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str

    # --- Параметры RAG ---
    chunk_size: int
    chunk_overlap: int
    top_k: int

    @property
    def db_dsn(self) -> str:
        """
        Собирает DSN (connection string) для psycopg.
        psycopg.connect() умеет принимать как кортеж параметров, так и DSN-строку.
        Строка удобнее, когда нужно её залогировать (с маскировкой пароля)
        или прокинуть в утилиту psql.
        """
        return (
            f"host={self.db_host} port={self.db_port} "
            f"dbname={self.db_name} user={self.db_user} "
            f"password={self.db_password}"
        )


def load_settings() -> Settings:
    """
    Создаёт Settings, читая значения из окружения.
    Вызывается один раз при импорте модуля — см. ниже.
    """
    return Settings(
        lm_studio_base_url=_env("LM_STUDIO_BASE_URL"),
        lm_studio_api_key=_env("LM_STUDIO_API_KEY"),
        embedding_model=_env("EMBEDDING_MODEL", "text-embedding-bge-m3"),
        embedding_dim=_env_int("EMBEDDING_DIM", 1024),
        chat_model=_env("CHAT_MODEL", "google/gemma-3-4b-it"),
        db_host=_env("DB_HOST", "localhost"),
        db_port=_env_int("DB_PORT", 5432),
        db_name=_env("DB_NAME", "rag"),
        db_user=_env("DB_USER", "rag"),
        db_password=_env("DB_PASSWORD", "rag"),
        chunk_size=_env_int("CHUNK_SIZE", 500),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 80),
        top_k=_env_int("TOP_K", 5),
    )


# Глобальный объект настроек. Импортируется во всех модулях как
# `from rag.config import settings`. Создаётся ровно один раз при
# первом импорте config.py.
settings = load_settings()
