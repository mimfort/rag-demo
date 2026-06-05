"""
Сборка LangGraph: clarify-фаза + ReAct-цикл.

Структура:
    START → interpret → confirm ──(да)──→ agent ⇄ tools
                ↑                  │          │
                └──────(нет)───────┘          ▼ (нет tool_call'ов)
                                            verify ──(ok)──→ END
                                              │
                                              └──(расхождение)──→ agent

- interpret: один LLM-вызов, перефразирует запрос (резолвит даты).
- confirm:   interrupt() — пауза до подтверждения пользователя.
- agent/tools: прежний ReAct-цикл, работает по подтверждённой формулировке.
- verify:    самопроверка финального ответа против данных tool'ов
             (anti-hallucination); при расхождении — возврат в agent.
"""

from __future__ import annotations

from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

from agent.clarify import interpret_query
from agent.config import MAX_CLARIFY_ROUNDS, MAX_VERIFY_ROUNDS
from agent.llm import make_llm
from agent.state import AgentState
from agent.tools import TOOLS
from agent.verify import verify_answer


SYSTEM_PROMPT_TEMPLATE = """Ты — помощник по СКК «Рондо». Ты можешь вызывать tools чтобы получить погоду и список забронированных кортов на конкретные даты.

Сегодня: {today}.

Если пользователь спрашивает про «свободный день», «следующую неделю», «погоду» — определи диапазон дат и проверь их tool'ами. Когда соберёшь достаточно данных — отвечай по-русски, упоминай конкретные даты и часы.

Отвечай КРАТКО и по делу: только нужные факты (даты, часы, температура, свободно/занято) и короткая рекомендация в 1–2 предложения. Без длинных вступлений и «воды», минимум форматирования и эмодзи. Не пересказывай весь список занятых часов, если хватает вывода «свободно/занято». Цель — дать ответ минимумом текста.

Не вызывай один и тот же tool с одинаковыми аргументами повторно. Когда ответ можно дать — отвечай без новых tool_call'ов."""


def _build_system_message() -> SystemMessage:
    return SystemMessage(SYSTEM_PROMPT_TEMPLATE.format(today=date.today().isoformat()))


def _original_query(state: AgentState) -> str:
    """Контент первого human-сообщения — исходный запрос пользователя."""
    for m in state["messages"]:
        if m.type == "human":
            return m.content
    return ""


def _final_answer_text(state: AgentState) -> str:
    """Текст последнего AIMessage без tool_call'ов — финальный ответ агента."""
    for m in reversed(state["messages"]):
        if m.type == "ai" and not getattr(m, "tool_calls", None):
            return m.content or ""
    return ""


def _verify_feedback_text(issue: str) -> str:
    """Корректирующий human-ход для агента после расхождения verify.

    Важно: переписка agent↔verify пользователю не видна, поэтому требуем
    выдать ТОЛЬКО исправленный ответ без мета-вступлений («вы правы»,
    «исправляю» и т.п.) — иначе они утекают в финальный ответ.
    """
    return (
        "Самопроверка нашла расхождение между твоим ответом и данными "
        f"инструментов: {issue}. Перепиши ответ строго по полученным данным — "
        "не добавляй температуру, даты, часы и слоты, которых нет в "
        "результатах инструментов. ВАЖНО: выдай только сам исправленный ответ "
        "для пользователя, без вступлений, извинений и фраз вроде «вы правы» "
        "или «исправляю» — пользователь не видит эту переписку."
    )


def _collect_tool_results(state: AgentState) -> list[dict]:
    """Все ToolMessage turn'а — сырьё, против которого верифицируем ответ."""
    return [
        {"name": getattr(m, "name", "") or "", "content": m.content}
        for m in state["messages"]
        if m.type == "tool"
    ]


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

    async def verify_node(state: AgentState) -> dict:
        rounds = state.get("verify_rounds", 0)
        tool_results = _collect_tool_results(state)
        # Нечего верифицировать (агент ответил без tool'ов) или исчерпан лимит
        # доработок — пропускаем проверку, отдаём ответ как есть.
        if not tool_results or rounds >= MAX_VERIFY_ROUNDS:
            return {"verify_ok": True, "verify_issue": None}

        query = state.get("effective_query") or _original_query(state)
        answer = _final_answer_text(state)
        ok, issue = await verify_answer(plain_llm, query, tool_results, answer)
        if ok:
            return {"verify_ok": True, "verify_issue": None}

        # Расхождение: добавляем корректирующий human-ход и растим счётчик.
        # route_after_verify уведёт обратно в agent — он перепишет ответ с
        # этим фидбеком в контексте. Счётчик не даст зациклиться.
        feedback = HumanMessage(_verify_feedback_text(issue))
        return {
            "verify_ok": False,
            "verify_issue": issue,
            "verify_rounds": rounds + 1,
            "messages": [feedback],
        }

    def route_after_verify(state: AgentState) -> str:
        return END if state.get("verify_ok") else "agent"

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("interpret", interpret_node)
    graph.add_node("confirm", confirm_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("verify", verify_node)

    graph.set_entry_point("interpret")
    graph.add_edge("interpret", "confirm")
    graph.add_conditional_edges(
        "confirm", route_after_confirm,
        {"agent": "agent", "interpret": "interpret"},
    )
    # tools_condition даёт "tools" (есть tool_call'ы) либо END (готов ответ).
    # Ветку END подменяем на verify — самопроверку перед выдачей.
    graph.add_conditional_edges(
        "agent", tools_condition,
        {"tools": "tools", END: "verify"},
    )
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "verify", route_after_verify,
        {"agent": "agent", END: END},
    )
    return graph.compile(checkpointer=checkpointer)
