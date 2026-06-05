from rag.generator import build_headers


def test_build_headers_minimal():
    h = build_headers("sk-or-x")
    assert h["Authorization"] == "Bearer sk-or-x"
    assert h["Content-Type"] == "application/json"
    assert "HTTP-Referer" not in h
    assert "X-Title" not in h


def test_build_headers_with_attribution():
    h = build_headers("sk-or-x", referer="https://app.test", title="RAG")
    assert h["HTTP-Referer"] == "https://app.test"
    assert h["X-Title"] == "RAG"
