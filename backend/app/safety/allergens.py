"""Allergen choices offered by the questionnaire; values are canonical dictionary groups."""

from __future__ import annotations

ALLERGEN_OPTIONS: tuple[dict[str, str], ...] = (
    {"value": "птиця", "label": "Птиця (курка, індичка, качка)"},
    {"value": "яловичина", "label": "Яловичина"},
    {"value": "червоне м'ясо", "label": "Червоне м'ясо (ягнятина, свинина)"},
    {"value": "риба", "label": "Риба й морепродукти"},
    {"value": "злаки з глютеном", "label": "Злаки з глютеном (пшениця, ячмінь, овес)"},
    {"value": "кукурудза", "label": "Кукурудза"},
    {"value": "соя", "label": "Соя"},
    {"value": "молочні продукти", "label": "Молочні продукти"},
    {"value": "яйця", "label": "Яйця"},
)
