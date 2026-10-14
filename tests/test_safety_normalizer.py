"""Regression tests for BUG-009/BUG-010: allergens from the questionnaire and free-form declarations."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "src"))

from petrec.composition import ingredient_list, is_safe

from app.safety import SafetyEvaluator
from app.safety.allergens import ALLERGEN_OPTIONS

evaluator = SafetyEvaluator()

QUESTIONNAIRE = [option["value"] for option in ALLERGEN_OPTIONS]

UNSAFE = [
    # (declaration, allergen from questionnaire or legacy profile value)
    ("chicken, rice, minerals", "курятина"),
    ("chicken, rice", "птиця"),
    ("dehydrated poultry protein, rice", "птиця"),
    ("dehydrated poultry protein (30%), maize", "курятина"),
    ("Chicken Meal 26%, brown rice", "курятина"),
    ("hydrolysed poultry liver, potato", "птиця"),
    ("turkey, peas", "птиця"),
    ("duck (20%), sweet potato", "птиця"),
    ("курка 30%, рис", "птиця"),
    ("м'ясо птиці, кукурудза", "птиця"),
    ("м’ясо та тваринні похідні (4% курки), злаки", "птиця"),
    ("meat and animal derivatives (4% chicken), cereals", "курятина"),
    ("курячий жир, горох", "птиця"),
    ("індичка, батат", "курятина"),
    ("wheat, chicken", "пшениця"),
    ("wheat flour, salmon", "пшениця"),
    ("пшеничне борошно, лосось", "пшениця"),
    ("barley, fish", "злаки з глютеном"),
    ("вівсянка, яловичина", "злаки з глютеном"),
    ("beef (18%), rice", "яловичина"),
    ("яловичина, рис", "яловичина"),
    ("salmon, potato", "риба"),
    ("лосось 40%, картопля", "риба"),
    ("herring meal, peas", "риба"),
    ("anchovy oil, rice", "риба"),
    ("milk powder, rice", "молочні продукти"),
    ("сухе молоко, рис", "молочні продукти"),
    ("whey protein, rice", "молочні продукти"),
    ("dried egg, rice", "яйця"),
    ("soybean meal, corn", "соя"),
]

SAFE = [
    ("salmon, rice, peas", "птиця"),
    ("лосось, рис, горох", "курятина"),
    ("duck, potato", "яловичина"),
    ("rice, peas, potato", "пшениця"),
    ("кролик, рис", "птиця"),
]


@pytest.mark.parametrize("declaration, allergen", UNSAFE)
def test_declarations_with_allergen_are_unsafe(declaration, allergen):
    decision = evaluator.evaluate(declaration, [allergen])
    assert decision.safe is False, (declaration, allergen, decision.items)
    assert decision.reason


@pytest.mark.parametrize("declaration, allergen", SAFE)
def test_declarations_without_allergen_stay_safe(declaration, allergen):
    assert evaluator.evaluate(declaration, [allergen]).safe is True


def test_core_alone_reproduced_bug_009_before_dictionary_fix():
    # The questionnaire value is now a dictionary variant, so the unchanged core also catches it.
    assert is_safe(ingredient_list("chicken, rice"), ["курятина"]) is False
    assert is_safe(ingredient_list("wheat, rice"), ["пшениця"]) is False


def test_every_questionnaire_option_maps_to_a_known_group():
    known = set(evaluator.normalizer.groups)
    for value in QUESTIONNAIRE:
        assert set(evaluator.allergen_groups([value])) <= known, value


def test_species_mismatch_is_unsafe():
    decision = evaluator.evaluate("salmon, rice", [], pet_species="cat", product_species="dog")
    assert decision.safe is False and "іншого виду" in decision.reason


def test_unspecified_animal_derivatives_stay_risky_for_allergic_pets():
    assert evaluator.evaluate("meat and animal derivatives, cereals", ["риба"]).safe is False
    assert evaluator.evaluate("meat and animal derivatives, cereals", []).safe is True


def test_specific_meat_source_does_not_add_unspecified_risk():
    assert evaluator.evaluate("м'ясо качки, рис", ["яловичина"]).safe is True


def test_unknown_parts_are_reported_for_dictionary_review():
    assert "chondroitin" in evaluator.normalizer.unknown_parts("chicken, chondroitin")
