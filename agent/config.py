"""
Настройки агент-модуля. Большинство значений живёт в .env (через
python-dotenv в rag/config.py), небольшие технические константы — здесь.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


# Защитный лимит итераций ReAct-loop (agent → tools → agent → ...).
# При превышении runner emit'нёт error и завершит граф.
MAX_ITER: int = 10

# Гард против бесконечного цикла подтверждения (interpret ⇄ confirm).
# После стольких кругов «нет» агент принудительно принимает последнюю
# интерпретацию и идёт работать.
MAX_CLARIFY_ROUNDS: int = 5

# Сколько раз verify-нода вправе вернуть агента на доработку ответа.
# После лимита самопроверка пропускается — отдаём, что есть, без зацикливания.
MAX_VERIFY_ROUNDS: int = 1

# Таймаут httpx-вызовов к skkrondo API (одного запроса).
HTTP_TIMEOUT_SEC: float = 10.0


@dataclass(frozen=True)
class AgentSettings:
    skkrondo_base_url: str

    @property
    def is_configured(self) -> bool:
        return bool(self.skkrondo_base_url)


def load_agent_settings() -> AgentSettings:
    return AgentSettings(
        skkrondo_base_url=(
            os.getenv("SKKRONDO_BASE_URL") or "https://api.skkrondo.ru"
        ).rstrip("/"),
    )


agent_settings = load_agent_settings()
