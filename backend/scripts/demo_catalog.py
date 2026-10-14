"""Deterministic demo catalog for PetMind (no research data needed).

Products are generated from protein × product line so the catalog exercises species filtering,
allergen safety (including hard declarations such as «dehydrated poultry protein» or
«meat and animal derivatives (4% chicken)»), sterilized-pet needs and wet/dry replenishment.
All goods are clearly marked as demo items with the «PetMind Demo» brand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BRAND = "PetMind Demo"
NOTE = "Демонстраційний товар PetMind; дані не є комерційною пропозицією."


@dataclass(frozen=True)
class Protein:
    key: str
    ua_with: str  # «з лососем»
    dry: str  # ingredient declaration of dry food
    wet: str
    image: dict[str, str]


CAT_PROTEINS = (
    Protein("salmon", "з лососем", "salmon (30%), rice, salmon oil, peas, beet pulp, minerals",
            "salmon (45%), fish broth, rice, minerals", {"cat": "cat-salmon", "dog": "dog-salmon"}),
    Protein("tuna", "з тунцем", "tuna (28%), potato, peas, fish oil, cellulose, taurine",
            "tuna (50%), fish broth, minerals, taurine", {"cat": "cat-salmon", "dog": "dog-salmon"}),
    Protein("whitefish", "з білою рибою", "whitefish (26%), sweet potato, peas, flaxseed, minerals",
            "cod (40%), whitefish, pumpkin, minerals", {"cat": "cat-salmon", "dog": "dog-salmon"}),
    Protein("beef", "з яловичиною", "beef (27%), potato, peas, sunflower oil, minerals",
            "beef (48%), beef broth, carrots, minerals", {"cat": "cat-chicken", "dog": "dog-beef"}),
    Protein("chicken", "з куркою", "chicken (30%), rice, chicken fat, corn, minerals",
            "м'ясо та тваринні похідні (4% курки), злаки, мінерали", {"cat": "cat-chicken", "dog": "dog-beef"}),
    Protein("turkey", "з індичкою", "dehydrated poultry protein (25%), turkey, maize, rice, animal fat",
            "turkey (42%), poultry broth, rice, minerals", {"cat": "cat-chicken", "dog": "dog-beef"}),
)
DOG_PROTEINS = (
    CAT_PROTEINS[0],
    Protein("lamb", "з ягнятиною", "lamb (26%), rice, barley, sunflower oil, minerals",
            "lamb (45%), lamb broth, potato, minerals", {"dog": "dog-beef"}),
    CAT_PROTEINS[3],
    Protein("duck", "з качкою", "duck (25%), sweet potato, peas, salmon oil, minerals",
            "duck (40%), duck broth, pumpkin, minerals", {"dog": "dog-beef"}),
    Protein("pork", "зі свининою", "pork (24%), wheat flour, corn, pork fat, minerals",
            "pork (44%), pork broth, carrots, minerals", {"dog": "dog-beef"}),
    CAT_PROTEINS[4],
)


@dataclass(frozen=True)
class Line:
    key: str
    title: str  # «Сухий корм для стерилізованих котів {protein}»
    category: str
    wet: bool
    protein: float
    fat: float
    moisture: float
    weight_kg: float
    base_price: float


CAT_LINES = (
    Line("adult", "Сухий корм для дорослих котів {p}", "Сухий корм", False, 32, 15, 9, 2.0, 589),
    Line("sterilized", "Сухий корм для стерилізованих котів {p}", "Сухий корм для стерилізованих котів",
         False, 35, 11, 9, 2.0, 629),
    Line("kitten", "Сухий корм для кошенят {p}", "Сухий корм для кошенят", False, 36, 18, 9, 1.5, 549),
    Line("wet", "Вологий корм для котів {p}, пауч", "Вологий корм", True, 10, 5, 80, 0.085, 39),
)
DOG_LINES = (
    Line("adult", "Сухий корм для дорослих собак {p}", "Сухий корм", False, 26, 14, 10, 3.0, 749),
    Line("puppy", "Сухий корм для цуценят {p}", "Сухий корм для цуценят", False, 30, 17, 10, 3.0, 789),
    Line("wet", "Вологий корм для собак {p}, консерва", "Вологий корм", True, 9, 6, 78, 0.4, 109),
)


def _row(species: str, protein: Protein, line: Line, index: int) -> dict[str, Any]:
    who = "котів" if species == "cat" else "собак"
    product_id = f"PETMIND-DEMO-{species.upper()}-{line.key.upper()}-{protein.key.upper()}"
    ingredients = protein.wet if line.wet else protein.dry
    form = "вологий" if line.wet else "сухий"
    return {
        "id": product_id,
        "title": line.title.format(p=protein.ua_with),
        "brand": BRAND,
        "category": line.category,
        "species": species,
        "price": round(line.base_price + 20 * (index % 4), 2),
        "image_url": f"/demo-products/{protein.image.get(species, 'cat-salmon')}.svg",
        "description": (f"Повнораціонний {form} корм для {who} {protein.ua_with}. "
                        f"Лінія «{line.category}». {NOTE}"),
        "ingredients_text": ingredients,
        "protein": float(line.protein),
        "fat": float(line.fat),
        "moisture": float(line.moisture),
        "is_regular": True,
        "package_weight_kg": line.weight_kg,
    }


def demo_catalog() -> list[dict[str, Any]]:
    rows = []
    for species, proteins, lines in (("cat", CAT_PROTEINS, CAT_LINES), ("dog", DOG_PROTEINS, DOG_LINES)):
        for p_index, protein in enumerate(proteins):
            for line in lines:
                rows.append(_row(species, protein, line, p_index))
    return rows
