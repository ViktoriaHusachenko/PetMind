import pandas as pd

from petrec.llm import MockLLM
from petrec.rerank import rerank_candidates, rerank_recommendations


def _candidates():
    return pd.DataFrame([
        {"item_id": "a", "s_cand": 0.9, "S_comp": 0.8, "safe": True},
        {"item_id": "b", "s_cand": 0.7, "S_comp": 0.6, "safe": True},
        {"item_id": "unsafe", "s_cand": 1.0, "S_comp": 1.0, "safe": False},
    ])


def test_valid_mock_response_and_safe_filter():
    llm = MockLLM('{"items":[{"id":"a","score":5,"reason":"Відповідає критеріям"},{"id":"b","score":3,"reason":"Підходить"}]}')
    result = rerank_candidates(_candidates(), llm, item_embeddings={"a": [1, 0], "b": [0, 1]})
    assert set(result.item_id) == {"a", "b"}
    assert result.iloc[0].item_id == "a"
    assert result.iloc[0].s_LLM == 1
    assert not result.llm_fallback.any()


def test_invalid_json_retries_once_then_uses_fallback():
    llm = MockLLM(["not json", "still not json"])
    result = rerank_candidates(_candidates(), llm)
    assert len(llm.calls) == 2
    assert result.llm_fallback.all()
    assert (result.s_LLM == 0.5).all()
    assert set(result.item_id) == {"a", "b"}


def test_hallucinated_id_retries_and_falls_back_if_repeated():
    invalid = '{"items":[{"id":"ghost","score":5,"reason":"Невідомий товар"}]}'
    llm = MockLLM([invalid, invalid])
    result = rerank_candidates(_candidates(), llm)
    assert len(llm.calls) == 2
    assert "ghost" not in set(result.item_id)
    assert result.llm_fallback.all()


def test_llm_cannot_return_or_bless_unsafe_item_and_ui_contract_is_stable():
    response = '{"items":[{"id":"unsafe","score":5,"reason":"Ідеальний товар"},{"id":"a","score":4,"reason":"Добрий вибір"}]}'
    corrected = '{"items":[{"id":"a","score":4,"reason":"Добрий вибір"},{"id":"b","score":3,"reason":"Підходить"}]}'
    llm = MockLLM([response, corrected])
    result = rerank_recommendations(_candidates(), llm)
    assert len(llm.calls) == 2
    assert {item.item_id for item in result} == {"a", "b"}
    assert all(item.safe_badge for item in result)
    assert result[0].match_percent == 82
    assert result[0].llm_used is True
    assert result[0].reason == "Добрий вибір"


def test_fallback_has_template_reason_and_ui_fields():
    result = rerank_recommendations(_candidates(), MockLLM(["bad", "bad"]))
    assert result
    assert all(item.llm_used is False for item in result)
    assert all("Відповідність складу" in item.reason for item in result)
    assert all(isinstance(item.composition_summary, list) for item in result)
