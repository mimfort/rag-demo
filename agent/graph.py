"""
Сборка LangGraph: clarify-фаза + ReAct-цикл.

Структура:
    START → interpret → confirm ──(да)──→ agent ⇄ tools → END
                ↑                  │
                └──────(нет)───────┘

- interpret: один LLM-вызов, перефразирует запрос (резолвит даты).
- confirm:   interrupt() — пауза до подтверждения пользователя.
- agent/tools: прежний ReAct-цикл, работает по подтверждённой формулировке.
"""

from __future__ import annotations

from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

from agent.clarify import interpret_query
from agent.config import MAX_CLARIFY_ROUNDS
from agent.llm import make_llm
from agent.state import AgentState
from agent.tools import TOOLS


SYSTEM_PROMPT_TEMPLATE = """Ты — помощник по СКК «Рондо». Ты можешь вызывать tools чтобы получить погоду и список забронированных кортов на конкретные даты.

Сегодня: {today}.

Если пользователь спрашивает про «свободный день», «следующую неделю», «погоду» — определи диапазон дат и проверь их tool'ами. Когда соберёшь достаточно данных — отвечай по-русски, упоминай конкретные даты и часы.

Не вызывай один и тот же tool с одинаковыми аргументами повторно. Когда ответ можно дать — отвечай без новых tool_call'ов."""


def _build_system_message() -> SystemMessage:
    return SystemMessage(SYSTEM_PROMPT_TEMPLATE.format(today=date.today().isoformat()))


def _original_query(state: AgentState) -> str:
    """Контент первого human-сообщения — исходный запрос пользователя."""
    for m in state["messages"]:
        if m.type == "human":
            return m.content
    return ""


def build_graph(plain_llm=None, agent_llm=None, checkpointer=None):
    """Создаёт скомпилированный граф. Вызывается один раз на процесс.

    Параметры plain_llm/agent_llm/checkpointer — для инъекции в тестах;
    в проде остаются None и создаются реальные объекты.
    """
    plain_llm = plain_llm if plain_llm is not None else make_llm()
    agent_llm = agent_llm if agent_llm is not None else make_llm().bind_tools(TOOLS)
    checkpointer = checkpointer if checkpointer is not None else MemorySaver()

    async def interpret_node(state: AgentState) -> dict:
        interpretation = await interpret_query(
            plain_llm,
            _original_query(state),
            state.get("correction"),
            date.today().isoformat(),
        )
        return {
            "interpretation": interpretation,
            "clarify_rounds": state.get("clarify_rounds", 0) + 1,
        }

    def confirm_node(state: AgentState) -> dict:
        rounds = state.get("clarify_rounds", 0)
        # Авто-приём после лимита — иначе цикл «нет» бесконечен.
        if rounds >= MAX_CLARIFY_ROUNDS:
            return {"effective_query": state["interpretation"], "correction": None}
        # interrupt() ставит граф на паузу; value уходит клиенту, а на resume
        # сюда приходит payload из Command(resume=...).
        answer = interrupt({
            "interpretation": state["interpretation"],
            "original": _original_query(state),
            "round": rounds,
        })
        if answer.get("confirmed"):
            return {"effective_query": state["interpretation"], "correction": None}
        # effective_query=None явно — чтобы route_after_confirm точно увёл
        # обратно в interpret, даже если поле когда-то выставят спекулятивно.
        return {"correction": answer.get("correction") or "", "effective_query": None}

    def route_after_confirm(state: AgentState) -> str:
        return "agent" if state.get("effective_query") else "interpret"

    async def agent_node(state: AgentState) -> dict:
        messages = list(state["messages"])
        eff = state.get("effective_query")
        if eff:
            # Подменяем ПЕРВЫЙ human-ход на подтверждённую формулировку: она и
            # есть реальный запрос агента. Приписка отдельным SystemMessage не
            # работала — «живой» human-ход перевешивал её, и модель отвечала на
            # оригинал. Меняем только локальную копию; в state остаётся оригинал
            # (для UI/истории), tool-вызовы и их результаты сохраняются.
            replaced = False
            rebuilt = []
            for m in messages:
                if not replaced and m.type == "human":
                    rebuilt.append(HumanMessage(eff))
                    replaced = True
                else:
                    rebuilt.append(m)
            messages = rebuilt
        if not messages or messages[0].type != "system":
            messages = [_build_system_message()] + messages
        response = await agent_llm.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("interpret", interpret_node)
    graph.add_node("confirm", confirm_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("interpret")
    graph.add_edge("interpret", "confirm")
    graph.add_conditional_edges(
        "confirm", route_after_confirm,
        {"agent": "agent", "interpret": "interpret"},
    )
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
