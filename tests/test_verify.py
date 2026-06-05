"""
Тесты verify-контролёра: парсинг вердикта и наличие анти-false-positive
директив в промпте (якорь на summary, смещение к OK).
"""

from agent.verify import VERIFIER_PROMPT, build_verifier_messages, parse_verdict


def test_parse_ok():
    assert parse_verdict("OK") == (True, "")
    assert parse_verdict("  ok  ") == (True, "")


def test_parse_problem():
    ok, issue = parse_verdict("ПРОБЛЕМА: выдумана температура")
    assert ok is False
    assert issue == "выдумана температура"


def test_parse_empty_defaults_to_problem():
    ok, issue = parse_verdict("")
    assert ok is False
    assert issue  # непустое дефолтное сообщение


def test_prompt_anchors_on_summary_and_biases_to_ok():
    # Тюнинг против false-positive: читать summary, игнорировать reservations,
    # при сомнении — OK.
    assert "summary" in VERIFIER_PROMPT
    assert "reservations" in VERIFIER_PROMPT
    assert "сомневаешься" in VERIFIER_PROMPT.lower()


def test_build_messages_includes_data_and_answer():
    msgs = build_verifier_messages(
        "погода завтра?",
        [{"name": "get_weather", "content": '{"temperature_c": 27}'}],
        "Завтра +27 °C.",
    )
    human = msgs[-1].content
    assert "погода завтра?" in human
    assert "get_weather" in human
    assert "Завтра +27" in human
