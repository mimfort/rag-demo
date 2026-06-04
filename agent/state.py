"""
State-схема для LangGraph-агента. Ядро — `messages` (канонический ReAct-state
в LangGraph: вся история turn'а лежит в нём). Остальные поля обслуживают
clarify-фазу (interpret/confirm) и необязательны (total=False) во входном state.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # `add_messages` — встроенный reducer LangGraph: при обновлении state
    # из узла новые сообщения АППЕНДЯТСЯ к существующим, а не заменяют.
    # Это правильное поведение для ReAct: каждый узел добавляет 1+ message.
    messages: Annotated[list[BaseMessage], add_messages]
    # Последняя предложенная перефразировка запроса (узел interpret).
    interpretation: str
    # Подтверждённая формулировка — её использует agent_node.
    effective_query: str
    # Последнее уточнение пользователя на «нет» (для нового круга).
    correction: str | None
    # Счётчик кругов clarify — гард против бесконечного цикла.
    clarify_rounds: int
