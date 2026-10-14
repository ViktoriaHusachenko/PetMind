"""Ingredient and allergen normalization in front of the core safety equation.

The core (`petrec.composition`) matches ingredients to its dictionary by exact variant only, and its
tokenizer drops parentheses containing numbers ("animal derivatives (4% chicken)"). This adapter maps
free-form English/Ukrainian declarations onto dictionary variants first, so the unchanged core `Safe`
sees every allergen. Matching data lives in `data/` (dictionary + regex patterns), not in code.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

from petrec.composition import DEFAULT_DICTIONARY, UNSPECIFIED_RISK, Ingredient, load_dictionary

logger = logging.getLogger("petmind.safety")
DEFAULT_PATTERNS = DEFAULT_DICTIONARY.parent / "ingredient_patterns.csv"
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "`": "'", "´": "'"})
_PERCENT = re.compile(r"\b\d+(?:[.,]\d+)?\s*%")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class _Match:
    start: int
    end: int
    variant: str


def _clean(text: str) -> str:
    value = text.lower().translate(_APOSTROPHES)
    value = _PERCENT.sub(" ", value)
    return _SPACES.sub(" ", value).strip(" \t\r\n.,;:-()[]")


class IngredientNormalizer:
    """Maps declaration parts and profile allergens to core dictionary variants and groups."""

    def __init__(self, dictionary: Mapping[str, Ingredient], patterns: Iterable[tuple[str, str]]) -> None:
        self.dictionary = dict(dictionary)
        rules: list[tuple[re.Pattern[str], str]] = []
        # Every dictionary variant matches as a whole phrase; curated stems cover Ukrainian forms.
        for variant in sorted(self.dictionary, key=len, reverse=True):
            rules.append((re.compile(r"(?<!\w)" + re.escape(variant) + r"(?!\w)"), variant))
        for pattern, variant in patterns:
            if variant not in self.dictionary:
                raise ValueError(f"pattern {pattern!r} maps to unknown dictionary variant {variant!r}")
            rules.append((re.compile(pattern, re.IGNORECASE), variant))
        self._rules = rules
        self.groups = sorted({item.allergen_group for item in self.dictionary.values()
                              if item.allergen_group and item.allergen_group != UNSPECIFIED_RISK})

    @classmethod
    def from_files(cls, dictionary_path: Path = DEFAULT_DICTIONARY, patterns_path: Path = DEFAULT_PATTERNS
                   ) -> "IngredientNormalizer":
        with Path(patterns_path).open(encoding="utf-8-sig", newline="") as stream:
            patterns = [(row["pattern"], row["variant"].strip().lower()) for row in csv.DictReader(stream)]
        return cls(load_dictionary(dictionary_path), patterns)

    def _group(self, variant: str) -> str:
        return self.dictionary[variant].allergen_group

    def variants_in(self, part: str) -> list[str]:
        """Dictionary variants found in one declaration part, longest match first, in text order."""
        text = _clean(part)
        if not text:
            return []
        if text in self.dictionary:
            return [text]
        found: list[_Match] = []
        for rule, variant in self._rules:
            for match in rule.finditer(text):
                if not any(m.start < match.end() and match.start() < m.end for m in found):
                    found.append(_Match(match.start(), match.end(), variant))
        found.sort(key=lambda m: m.start)
        variants = list(dict.fromkeys(m.variant for m in found))
        specific = [v for v in variants if self._group(v) and self._group(v) != UNSPECIFIED_RISK]
        if specific:
            # "м'ясо птиці" names its source: the generic meat/derivative rule must not add a risk flag.
            variants = [v for v in variants if self._group(v) != UNSPECIFIED_RISK]
        return variants

    def parts(self, text: str | None) -> list[str]:
        """Split a declaration at commas/semicolons and also look inside parentheses."""
        if not text:
            return []
        value = str(text).translate(_APOSTROPHES)
        pieces: list[str] = []
        for part in re.split(r"[,;]", value):
            outer = re.sub(r"\([^)]*\)", " ", part)
            pieces.append(outer)
            pieces.extend(re.findall(r"\(([^)]*)\)", part))
        return [piece for piece in (p.strip() for p in pieces) if _clean(piece)]

    def canonical_text(self, text: str | None) -> str:
        """Declaration rewritten with dictionary variants (unknown parts kept) for the core parser."""
        tokens: list[str] = []
        for part in self.parts(text):
            variants = self.variants_in(part)
            tokens.extend(variants if variants else [_clean(part)])
        return ", ".join(dict.fromkeys(token for token in tokens if token))

    def unknown_parts(self, text: str | None) -> list[str]:
        return [_clean(part) for part in self.parts(text) if not self.variants_in(part)]

    def allergen_groups(self, allergens: Iterable[str] | None) -> list[str]:
        """Profile allergens ("курятина", "пшениця", "chicken") → canonical allergen groups."""
        groups: list[str] = []
        for raw in allergens or []:
            value = _clean(str(raw))
            if not value:
                continue
            if value in self.groups:
                groups.append(value)
                continue
            matched = [self._group(v) for v in self.variants_in(value) if self._group(v)]
            matched = [g for g in matched if g != UNSPECIFIED_RISK]
            if matched:
                groups.extend(matched)
            else:
                # Keep the raw value so it is never silently dropped; it can still match a group name.
                logger.warning("Unknown allergen in pet profile: %r", raw)
                groups.append(value)
        return list(dict.fromkeys(groups))


@lru_cache(maxsize=1)
def default_normalizer() -> IngredientNormalizer:
    return IngredientNormalizer.from_files()
