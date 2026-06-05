"""
Тест корректирующего фидбэка verify → agent.

Регрессия: фидбэк должен требовать выдать ТОЛЬКО исправленный ответ без
мета-вступлений («вы правы», «исправляю»), иначе они утекают в финальный
ответ пользователю.
"""

from agent.graph import _verify_feedback_text


def test_feedback_includes_issue():
    t = _verify_feedback_text("корт 2 занят в 10:00")
    assert "корт 2 занят в 10:00" in t


def test_feedback_forbids_meta_preamble():
    t = _verify_feedback_text("issue").lower()
    # Требуем «только исправленный ответ» и явно запрещаем фразы-извинения.
    assert "только" in t
    assert "вы правы" in t
    assert "исправляю" in t
    assert "пользователь не видит" in t
