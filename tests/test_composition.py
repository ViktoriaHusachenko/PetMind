from petrec.composition import composition_score, ingredient_list, load_dictionary, tokenize


def test_tokenize_removes_percentages_and_numeric_parentheses():
    assert tokenize("Chicken (25%), wheat 10%; rice (12)") == ["chicken", "wheat", "rice"]


def test_cat_with_poultry_allergy_and_chicken_first_is_unsafe():
    result = composition_score(
        "chicken, rice", allergens=["птиця"], nutrients={"protein": 30, "fat": 15}, moisture=10
    )
    assert result["safe"] is False
    assert result["S_comp"] == 0


def test_unspecified_animal_derivatives_are_risky_for_any_allergy():
    result = composition_score(
        "meat and animal derivatives", allergens=["риба"], nutrients={"protein": 30, "fat": 15}
    )
    assert result["safe"] is False
    assert result["S_comp"] == 0


def test_missing_nutrients_are_flagged_and_unknown_term_retained():
    items = ingredient_list("unlisted ingredient", load_dictionary())
    assert items == ["unlisted ingredient"]
    result = composition_score("rice", nutrients=None)
    assert result["S_nutr"] == 0.5
    assert result["nutrient_data_missing"] is True
