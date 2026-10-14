"""The single deterministic safety decision used by catalog, recommendations, cart and agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from petrec.composition import Ingredient, composition_score, ingredient_list, is_safe, safety_explanation

from .ingredients import IngredientNormalizer, default_normalizer

SPECIES_MISMATCH = "Товар призначений для іншого виду тварин."


@dataclass(frozen=True)
class SafetyDecision:
    safe: bool
    reason: str | None
    items: list[Ingredient | str]
    allergen_groups: list[str]


class SafetyEvaluator:
    """Normalizes inputs, then applies the unchanged core `Safe` equation. Never consults an LLM."""

    def __init__(self, normalizer: IngredientNormalizer | None = None) -> None:
        self.normalizer = normalizer or default_normalizer()

    def items(self, ingredients_text: str | None) -> list[Ingredient | str]:
        return ingredient_list(self.normalizer.canonical_text(ingredients_text), self.normalizer.dictionary)

    def allergen_groups(self, allergens: Iterable[str] | None) -> list[str]:
        return self.normalizer.allergen_groups(allergens)

    def evaluate(self, ingredients_text: str | None, allergens: Iterable[str] | None, *,
                 pet_species: str | None = None, product_species: str | None = None) -> SafetyDecision:
        items = self.items(ingredients_text)
        groups = self.allergen_groups(allergens)
        if pet_species and product_species and pet_species != product_species:
            return SafetyDecision(False, SPECIES_MISMATCH, items, groups)
        safe = bool(is_safe(items, groups))
        reason = None if safe else (safety_explanation(items, groups) or "Склад не пройшов перевірку безпечності.")
        return SafetyDecision(safe, reason, items, groups)

    def is_allergen(self, item: Ingredient | str, allergens: Iterable[str] | None) -> bool:
        return bool(safety_explanation([item], self.allergen_groups(allergens)))

    def composition(self, ingredients_text: str | None, allergens: Iterable[str] | None, **kwargs: Any) -> dict[str, Any]:
        return composition_score(self.items(ingredients_text), allergens=self.allergen_groups(allergens), **kwargs)


_default: SafetyEvaluator | None = None


def default_evaluator() -> SafetyEvaluator:
    global _default
    if _default is None:
        _default = SafetyEvaluator()
    return _default
