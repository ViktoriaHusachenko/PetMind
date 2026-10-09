import pandas as pd

from petrec.candidates import generate_candidates, item_cosine


def test_binary_item_cosine_and_safe_recent_purchase_filters():
    interactions = pd.DataFrame([
        {"user_id": "u1", "item_id": "a", "rating": 1},
        {"user_id": "u1", "item_id": "b", "rating": 1},
        {"user_id": "u1", "item_id": "c", "rating": 1},
        {"user_id": "u2", "item_id": "a", "rating": 1},
        {"user_id": "u2", "item_id": "b", "rating": 1},
        {"user_id": "u3", "item_id": "a", "rating": 1},
    ])
    cosine = item_cosine(interactions)
    assert cosine.loc["a", "b"] > cosine.loc["a", "c"]
    result = generate_candidates(
        "u1", interactions, ["c", "d", "unsafe", "recent", "regular"],
        {"c": [1, 0], "d": [0, 1], "unsafe": [1, 1], "recent": [1, 0], "regular": [0.5, 0.5]},
        [1, 0], safe={"c": True, "d": True, "unsafe": False, "recent": True, "regular": True},
        purchases=pd.DataFrame([
            {"user_id": "u1", "item_id": "recent", "timestamp": "2026-10-01"},
            {"user_id": "u1", "item_id": "regular", "timestamp": "2026-10-01"},
        ]), regular_items=["regular"], now="2026-10-02", top_n=50,
    )
    assert set(result.item_id) == {"c", "d", "regular"}
    assert "unsafe" not in set(result.item_id)
    assert result.safe.all()
    assert list(result.columns) == ["item_id", "s_cf", "s_content", "s_cand", "safe"]


def test_small_history_disables_cf_weight():
    interactions = pd.DataFrame([{"user_id": "u", "item_id": item} for item in ["a", "b"]])
    result = generate_candidates("u", interactions, ["x", "y"],
                                 {"x": [1, 0], "y": [0, 1]}, [1, 0], safe={"x": True, "y": True})
    assert result.iloc[0].item_id == "x"
    assert result.iloc[0].s_cand == 1
