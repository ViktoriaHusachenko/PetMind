"""Token-based product search shared by the catalog and the agent's search_products tool.

A phrase such as «корм для стерилізованого кота» is split into stems that tolerate Ukrainian
inflection («стерилізованого» → «стериліз»), so it matches «Корм для стерилізованих котів».
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

_WORD = re.compile(r"[\w']+", re.UNICODE)
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "`": "'"})
STOPWORDS = frozenset({
    "для", "і", "й", "та", "з", "із", "зі", "в", "у", "на", "по", "до", "від", "що", "як", "чи", "є", "мені",
    "мого", "моєї", "моєму", "моїй", "мій", "моя", "наш", "наша", "будь", "ласка", "покажи", "знайди",
    "підбери", "порадь", "хочу", "потрібен", "потрібна", "потрібно", "треба", "шукаю", "якийсь", "який",
    "яка", "які", "дай", "дайте", "можна", "а", "the", "for", "a", "an", "and", "with", "of", "my", "me",
    "show", "find", "need", "some",
})
# Species are filtered structurally by the pet profile, so they must not dominate text relevance.
SPECIES_WORDS = frozenset({
    "кіт", "кота", "коту", "котом", "коти", "котів", "котика", "котику", "кішка", "кішки", "кішці", "кошеня",
    "кошенят", "собака", "собаки", "собаці", "собакою", "собак", "пес", "пса", "псу", "цуценя", "цуценят",
    "cat", "cats", "kitten", "dog", "dogs", "puppy",
})
FIELD_WEIGHTS: Mapping[str, float] = {"title": 3.0, "category": 2.0, "brand": 2.0, "description": 1.0,
                                       "ingredients_text": 1.0}


def stem(word: str) -> str:
    """Prefix stem: drop up to three inflection letters, keep 4–8 characters."""
    word = word.lower().translate(_APOSTROPHES)
    if len(word) <= 4:
        return word
    return word[:min(8, max(4, len(word) - 3))]


@dataclass(frozen=True)
class SearchQuery:
    raw: str
    stems: tuple[str, ...]

    @classmethod
    def parse(cls, text: str | None) -> "SearchQuery":
        words = [w for w in _WORD.findall((text or "").lower().translate(_APOSTROPHES)) if len(w) >= 2]
        stems = [stem(w) for w in words if w not in STOPWORDS and w not in SPECIES_WORDS and not w.isdigit()]
        return cls(text or "", tuple(dict.fromkeys(stems)))

    @property
    def empty(self) -> bool:
        return not self.stems

    def score(self, fields: Mapping[str, str | None]) -> float:
        """Weighted share of query stems found in the product fields (0 = no match)."""
        if self.empty:
            return 0.0
        texts = {name: (value or "").lower().translate(_APOSTROPHES) for name, value in fields.items()}
        total = 0.0
        for item in self.stems:
            total += max((FIELD_WEIGHTS.get(name, 1.0) for name, text in texts.items() if item in text), default=0.0)
        return total / (len(self.stems) * max(FIELD_WEIGHTS.values()))


def rank(query: SearchQuery, rows: Iterable[tuple[str, Mapping[str, str | None]]], limit: int | None = None
         ) -> list[tuple[str, float]]:
    """(id, fields) rows → ids sorted by descending score; zero-score rows are dropped."""
    scored = [(row_id, query.score(fields)) for row_id, fields in rows]
    scored = sorted((item for item in scored if item[1] > 0), key=lambda item: (-item[1], item[0]))
    return scored[:limit] if limit else scored
