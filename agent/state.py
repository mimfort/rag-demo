"""
State-схема для LangGraph-агента. Только одно поле — `messages` — это
канонический ReAct-state в LangGraph (вся история turn'а лежит в нём).
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # `add_messages` — встроенный reducer LangGraph: при обновлении state
    # из узла новые сообщения АППЕНДЯТСЯ к существующим, а не заменяют.
    # Это правильное поведение для ReAct: каждый узел добавляет 1+ message.
    messages: Annotated[list[BaseMessage], add_messages]
