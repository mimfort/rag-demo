"""
Сборка LangGraph: ReAct-цикл agent ⇄ tools.

Структура:
    START
      ↓
    agent  ─── tool_calls? ──→ tools
      ↑                          │
      └──────────────────────────┘
      (если нет tool_calls → END)
"""

from __future__ import annotations

from datetime import date

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.llm import make_llm
from agent.state import AgentState
from agent.tools import TOOLS


SYSTEM_PROMPT_TEMPLATE = """Ты — помощник по СКК «Рондо». Ты можешь вызывать tools чтобы получить погоду и список забронированных кортов на конкретные даты.

Сегодня: {today}.

Если пользователь спрашивает про «свободный день», «следующую неделю», «погоду» — определи диапазон дат и проверь их tool'ами. Когда соберёшь достаточно данных — отвечай по-русски, упоминай конкретные даты и часы.

Не вызывай один и тот же tool с одинаковыми аргументами повторно. Когда ответ можно дать — отвечай без новых tool_call'ов."""


def _build_system_message() -> SystemMessage:
    return SystemMessage(
        SYSTEM_PROMPT_TEMPLATE.format(today=date.today().isoformat())
    )


def build_graph():
    """Создаёт скомпилированный граф. Вызывается один раз на процесс
    (heavy — bind_tools шлёт каждый раз tools-schema'у в LLM на инвоке).
    """
    llm_with_tools = make_llm().bind_tools(TOOLS)

    async def agent_node(state: AgentState) -> dict:
        """Вызов LLM. Если в state нет system-message — добавляем
        в начало (даты живые, не вшиты в build)."""
        messages = list(state["messages"])
        if not messages or messages[0].type != "system":
            messages = [_build_system_message()] + messages
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    # tools_condition возвращает "tools" если у последнего сообщения
    # есть .tool_calls, иначе END.
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()
