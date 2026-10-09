"""Ingredient normalization and transparent pet-food composition scoring."""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import config

DEFAULT_DICTIONARY = Path(__file__).resolve().parents[2] / "data" / "ingredient_normalization.csv"
UNSPECIFIED_RISK = "невизначена (ризик)"


@dataclass(frozen=True)
class Ingredient:
    variant: str
    canonical_name: str
    allergen_group: str
    theta: float


def tokenize(text: str | None) -> list[str]:
    """Lowercase and split an ingredient declaration at commas and semicolons."""
    if not text:
        return []
    value = str(text).lower()
    value = re.sub(r"\([^)]*\d[^)]*\)", " ", value)
    value = re.sub(r"\b\d+(?:[.,]\d+)?\s*%", " ", value)
    return [part.strip(" \t\r\n.,;:-") for part in re.split(r"[,;]", value) if part.strip(" \t\r\n.,;:-")]


def load_dictionary(path: str | Path = DEFAULT_DICTIONARY) -> dict[str, Ingredient]:
    """Read variant → canonical/group/theta mappings from a CSV file."""
    result: dict[str, Ingredient] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            variant = row["variant"].strip().lower()
            theta = float(row["theta"])
            if not 0 <= theta <= 1:
                raise ValueError(f"theta must be between 0 and 1 for {variant!r}")
            result[variant] = Ingredient(
                variant, row["canonical_name"].strip(), row.get("allergen_group", "").strip(), theta
            )
    return result


def build_dictionary_from_counter(
    counts: Counter[str] | Mapping[str, int], output_path: str | Path,
    suggestions: Mapping[str, Ingredient] | None = None, limit: int = 100,
) -> list[str]:
    """Write the 60–100 most frequent ingredients with supplied reviewed mappings.

    Unmapped frequent terms are written with themselves as the provisional name,
    blank group, and neutral theta, making the generated CSV ready for review.
    """
    if not 60 <= limit <= 100:
        raise ValueError("limit must be between 60 and 100")
    selected = [term.strip().lower() for term, _ in Counter(counts).most_common(limit) if term.strip()]
    lookup = suggestions or {}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "canonical_name", "allergen_group", "theta"])
        for term in selected:
            item = lookup.get(term)
            writer.writerow([term, item.canonical_name if item else term,
                             item.allergen_group if item else "", item.theta if item else 0.5])
    return selected


def ingredient_list(text: str | None, dictionary: Mapping[str, Ingredient] | None = None) -> list[Ingredient | str]:
    """Return ordered I_i = N(T(c_i)); unknown terms remain visible as strings."""
    mappings = dictionary if dictionary is not None else load_dictionary()
    return [mappings.get(token, token) for token in tokenize(text)]


def suggest_canonical(variant: str, llm_client: Any) -> dict[str, Any]:
    """Ask an LLM for a proposed mapping and return it for manual review only."""
    prompt = ("Suggest a canonical pet-food ingredient mapping as JSON with keys "
              "canonical_name, allergen_group, theta (0..1). Do not infer safety. "
              f"Ingredient: {variant}")
    if hasattr(llm_client, "complete"):
        proposal = llm_client.complete(prompt)
    elif hasattr(llm_client, "generate"):
        proposal = llm_client.generate(prompt)
    else:
        raise TypeError("llm_client must provide complete(prompt) or generate(prompt)")
    if isinstance(proposal, str):
        import json
        proposal = json.loads(proposal)
    if not isinstance(proposal, Mapping):
        raise ValueError("LLM proposal must be a mapping")
    theta = float(proposal.get("theta", 0.5))
    if not 0 <= theta <= 1:
        raise ValueError("proposed theta must be between 0 and 1")
    return {"variant": variant.strip().lower(), "canonical_name": str(proposal.get("canonical_name", variant)).strip(),
            "allergen_group": str(proposal.get("allergen_group", "")).strip(), "theta": theta,
            "status": "pending_manual_review"}


def _groups(items: Iterable[Ingredient | str]) -> set[str]:
    result = set()
    for item in items:
        if isinstance(item, Ingredient) and item.allergen_group:
            result.add(item.allergen_group)
    return result


def safety_explanation(items: Iterable[Ingredient | str], allergens: Iterable[str] = ()) -> str | None:
    """Return a deterministic Ukrainian explanation when known data marks an item unsafe."""
    groups = {group.lower() for group in _groups(items)}
    dictionary = load_dictionary()
    allergy_groups = set()
    for allergen in allergens:
        value = str(allergen).strip().lower()
        normalized = dictionary.get(value)
        allergy_groups.add(normalized.allergen_group.lower() if normalized and normalized.allergen_group else value)
    aliases = {"chicken": "птиця", "poultry": "птиця", "птицю": "птиця"}
    allergy_groups = {aliases.get(value, value) for value in allergy_groups}
    if UNSPECIFIED_RISK.lower() in groups and allergy_groups:
        return "Склад містить невизначені тваринні похідні за наявності алергії."
    matches = groups & allergy_groups
    if matches:
        return "Склад містить групу алергенів: " + ", ".join(sorted(matches)) + "."
    return None


def is_safe(
    items: Iterable[Ingredient | str], allergens: Iterable[str] = (), *,
    species: str | None = None, supported_species: Iterable[str] | None = None,
    age_group: str | None = None, supported_age_groups: Iterable[str] | None = None,
) -> bool:
    """Apply allergy, species and age suitability indicators in the safety equation."""
    listed = list(items)
    groups = {value.lower() for value in _groups(listed)}
    dictionary = load_dictionary()
    allergy_set = {
        dictionary[value.strip().lower()].allergen_group.lower()
        if value.strip().lower() in dictionary and dictionary[value.strip().lower()].allergen_group
        else value.strip().lower()
        for value in allergens
    }
    if UNSPECIFIED_RISK.lower() in groups and allergy_set:
        return False
    aliases = {"chicken": "птиця", "poultry": "птиця", "птицю": "птиця", "птиця": "птиця"}
    allergy_set = {aliases.get(value, value) for value in allergy_set}
    if groups & allergy_set:
        return False
    if species is not None and supported_species is not None and species not in supported_species:
        return False
    if age_group is not None and supported_age_groups is not None and age_group not in supported_age_groups:
        return False
    return True


def nutrient_score(
    nutrients: Mapping[str, float | None] | None, *, moisture: float | None = None,
    minimums: Mapping[str, float] | None = None,
) -> tuple[float, bool]:
    """Score protein/fat minimums on dry matter; return (score, missing-data flag)."""
    thresholds = dict(minimums or {
        "protein": config.FEDIAF_MIN_PROTEIN_DRY_MATTER,
        "fat": config.FEDIAF_MIN_FAT_DRY_MATTER,
    })
    nutrients = nutrients or {}
    if any(nutrients.get(key) is None for key in thresholds) or moisture is None:
        return 0.5, True
    water = float(moisture)
    if not 0 <= water < 100:
        raise ValueError("moisture must be in [0, 100)")
    scores = []
    for key, minimum in thresholds.items():
        dry = 100.0 * float(nutrients[key]) / (100.0 - water)
        scores.append(1.0 - min(1.0, max(0.0, float(minimum) - dry) / float(minimum)))
    return sum(scores) / len(scores), False


def ingredient_quality(items: Iterable[Ingredient | str], decay: float = config.COMPOSITION_DECAY) -> float:
    """Compute position-discounted Q_ingr from theta values."""
    if not 0 < decay <= 1:
        raise ValueError("decay must be in (0, 1]")
    values = list(items)
    if not values:
        return 0.0
    weights = [decay ** index for index in range(len(values))]
    scores = [item.theta if isinstance(item, Ingredient) else 0.0 for item in values]
    return sum(w * s for w, s in zip(weights, scores)) / sum(weights)


def composition_score(
    ingredients: str | Iterable[Ingredient | str], *, allergens: Iterable[str] = (),
    nutrients: Mapping[str, float | None] | None = None, moisture: float | None = None,
    species: str | None = None, supported_species: Iterable[str] | None = None,
    age_group: str | None = None, supported_age_groups: Iterable[str] | None = None,
    dictionary: Mapping[str, Ingredient] | None = None,
    minimums: Mapping[str, float] | None = None,
    eta: float = config.COMPOSITION_NUTRIENT_WEIGHT, decay: float = config.COMPOSITION_DECAY,
) -> dict[str, Any]:
    """Return Safe, S_nutr, Q_ingr and S_comp along with ordered normalized ingredients."""
    if not 0 <= eta <= 1:
        raise ValueError("eta must be between 0 and 1")
    items = ingredient_list(ingredients, dictionary) if isinstance(ingredients, str) else list(ingredients)
    safe = is_safe(items, allergens, species=species, supported_species=supported_species,
                   age_group=age_group, supported_age_groups=supported_age_groups)
    species_minimums = minimums
    if species_minimums is None and species in config.FEDIAF_MINIMUMS_DRY_MATTER:
        species_minimums = config.FEDIAF_MINIMUMS_DRY_MATTER[species]
    nutr, missing = nutrient_score(nutrients, moisture=moisture, minimums=species_minimums)
    quality = ingredient_quality(items, decay)
    score = float(safe) * (eta * nutr + (1 - eta) * quality)
    return {"safe": safe, "S_nutr": nutr, "nutrient_data_missing": missing,
            "Q_ingr": quality, "S_comp": score, "ingredients": items}
