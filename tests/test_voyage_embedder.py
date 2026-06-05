import httpx
import pytest

from rag.embedder import VoyageEmbedder, make_embedder, Embedder


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_embed_documents_parses_and_sorts():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["body"] = json.loads(request.content)
        # Намеренно перепутанный порядок index — клиент должен отсортировать.
        return httpx.Response(200, json={"data": [
            {"index": 1, "embedding": [0.1, 0.2, 0.3]},
            {"index": 0, "embedding": [0.4, 0.5, 0.6]},
        ]})

    emb = VoyageEmbedder(api_key="pa", model="voyage-4-large", dim=3,
                         client=_client(handler))
    vectors = emb.embed_documents(["a", "b"])

    assert vectors == [[0.4, 0.5, 0.6], [0.1, 0.2, 0.3]]
    assert captured["body"]["input_type"] == "document"
    assert captured["body"]["model"] == "voyage-4-large"


def test_embed_query_sets_input_type_query():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [
            {"index": 0, "embedding": [1.0, 2.0, 3.0]},
        ]})

    emb = VoyageEmbedder(api_key="pa", dim=3, client=_client(handler))
    vec = emb.embed_query("hello")

    assert vec == [1.0, 2.0, 3.0]
    assert captured["body"]["input_type"] == "query"
    assert captured["body"]["input"] == ["hello"]


def test_embed_dimension_mismatch_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [
            {"index": 0, "embedding": [1.0, 2.0]},  # длина 2, ждём 3
        ]})

    emb = VoyageEmbedder(api_key="pa", dim=3, client=_client(handler))
    with pytest.raises(RuntimeError):
        emb.embed_query("x")


def test_empty_documents_no_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("сеть не должна вызываться на пустом входе")

    emb = VoyageEmbedder(api_key="pa", dim=3, client=_client(handler))
    assert emb.embed_documents([]) == []


def test_make_embedder_returns_protocol(monkeypatch):
    monkeypatch.setattr("rag.embedder.settings",
                        type("S", (), {"voyage_base_url": "https://api.voyageai.com/v1",
                                       "voyage_api_key": "pa",
                                       "voyage_embedding_model": "voyage-4-large",
                                       "embedding_dim": 1024})())
    e = make_embedder()
    assert isinstance(e, Embedder)
