import httpx
import pytest

from rag.reranker import VoyageReranker, make_reranker, Reranker
from rag.vector_store import RetrievedChunk


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _chunk(content: str) -> RetrievedChunk:
    # Заполняем только обязательные поля; остальные — дефолтные/None.
    return RetrievedChunk(
        source="s", chunk_index=0, content=content, similarity=0.0,
    )


def test_rerank_maps_index_and_sorts():
    def handler(request: httpx.Request) -> httpx.Response:
        # Воспроизводим формат Voyage: index указывает на исходный документ.
        return httpx.Response(200, json={"data": [
            {"index": 0, "relevance_score": 0.2},
            {"index": 1, "relevance_score": 0.9},
        ]})

    rr = VoyageReranker(api_key="pa", model="rerank-2.5", client=_client(handler))
    chunks = [_chunk("low"), _chunk("high")]
    out = rr.rerank("q", chunks)

    assert [c.content for c in out] == ["high", "low"]
    assert out[0].reranker_score == 0.9
    assert out[0].original_rank == 2  # был вторым во входе (index 1 → rank 2)
    assert out[1].reranker_score == 0.2


def test_rerank_applies_top_k_in_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [
            {"index": 0, "relevance_score": 0.5},
        ]})

    rr = VoyageReranker(api_key="pa", client=_client(handler))
    rr.rerank("q", [_chunk("a"), _chunk("b")], top_k=1)
    assert captured["body"]["top_k"] == 1
    assert captured["body"]["query"] == "q"
    assert captured["body"]["documents"] == ["a", "b"]


def test_rerank_empty_no_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("сеть не должна вызываться на пустом входе")

    rr = VoyageReranker(api_key="pa", client=_client(handler))
    assert rr.rerank("q", []) == []


def test_make_reranker_returns_protocol(monkeypatch):
    monkeypatch.setattr("rag.reranker.settings",
                        type("S", (), {"voyage_base_url": "https://api.voyageai.com/v1",
                                       "voyage_api_key": "pa",
                                       "voyage_rerank_model": "rerank-2.5"})())
    assert isinstance(make_reranker(), Reranker)


def test_rerank_index_out_of_range_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        # index 5, а на входе только 1 документ → ответ некорректен.
        return httpx.Response(200, json={"data": [
            {"index": 5, "relevance_score": 0.9},
        ]})

    rr = VoyageReranker(api_key="pa", client=_client(handler))
    with pytest.raises(RuntimeError):
        rr.rerank("q", [_chunk("only")])


def test_rerank_missing_data_field_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list"})  # нет "data"

    rr = VoyageReranker(api_key="pa", client=_client(handler))
    with pytest.raises(RuntimeError):
        rr.rerank("q", [_chunk("x")])
