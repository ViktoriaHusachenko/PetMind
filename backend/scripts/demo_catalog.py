"""PetMind store catalog: house-brand food for every pet species and the vet pharmacy.

Rows are generated deterministically (protein × product line for cats and dogs, fixed lists for
other species and the pharmacy) so the seed is idempotent. Photos live in
`frontend/public/products/` (freely licensed, see `credits.json`). Ids keep the historical
`PETMIND-DEMO-` prefix so existing purchases and reminders stay linked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BRAND = "PetMind"
VET_NOTE = ("Перед застосуванням ознайомтеся з інструкцією та проконсультуйтеся з ветеринарним лікарем. "
            "Не перевищуйте рекомендовану кратність обробок.")
DRY_KCAL, WET_KCAL = 3800.0, 900.0
# Pack photos of real brands (Open Pet Food Facts, CC BY-SA): photo name -> brand shown on the card.
BRAND_PHOTOS: dict[str, str] = json.loads((Path(__file__).with_name("brand_photos.json")).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Protein:
    key: str
    ua_with: str  # «з лососем»
    dry: str
    wet: str
    note: str  # one sentence about the protein


CAT_PROTEINS = (
    Protein("salmon", "з лососем", "salmon (30%), rice, salmon oil, peas, beet pulp, minerals",
            "salmon (45%), fish broth, rice, minerals", "Лосось — джерело омега-3 для блиску шерсті та здоров'я шкіри."),
    Protein("tuna", "з тунцем", "tuna (28%), potato, peas, fish oil, cellulose, taurine",
            "tuna (50%), fish broth, minerals, taurine", "Тунець має виразний смак, який люблять навіть перебірливі коти."),
    Protein("whitefish", "з білою рибою", "whitefish (26%), sweet potato, peas, flaxseed, minerals",
            "cod (40%), whitefish, pumpkin, minerals", "Біла риба — легкозасвоюваний нежирний білок."),
    Protein("beef", "з яловичиною", "beef (27%), potato, peas, sunflower oil, minerals",
            "beef (48%), beef broth, carrots, minerals", "Яловичина — повноцінний тваринний білок для підтримки м'язів."),
    Protein("chicken", "з куркою", "chicken (30%), rice, chicken fat, corn, minerals",
            "м'ясо та тваринні похідні (4% курки), злаки, мінерали", "Курка — класичне джерело білка для щоденного раціону."),
    Protein("turkey", "з індичкою", "dehydrated poultry protein (25%), turkey, maize, rice, animal fat",
            "turkey (42%), poultry broth, rice, minerals", "Індичка — ніжне м'ясо з високим вмістом білка."),
)
DOG_PROTEINS = (
    CAT_PROTEINS[0],
    Protein("lamb", "з ягнятиною", "lamb (26%), rice, barley, sunflower oil, minerals",
            "lamb (45%), lamb broth, potato, minerals", "Ягнятина підходить собакам, чутливим до курятини."),
    CAT_PROTEINS[3],
    Protein("duck", "з качкою", "duck (25%), sweet potato, peas, salmon oil, minerals",
            "duck (40%), duck broth, pumpkin, minerals", "Качка з бататом — смачна альтернатива звичним білкам."),
    Protein("pork", "зі свининою", "pork (24%), wheat flour, corn, pork fat, minerals",
            "pork (44%), pork broth, carrots, minerals", "Свинина — енергійний раціон для активних собак."),
    CAT_PROTEINS[4],
)


@dataclass(frozen=True)
class Line:
    key: str
    title: str
    category: str
    wet: bool
    protein: float
    fat: float
    moisture: float
    weight_kg: float
    base_price: float
    images: tuple[str, ...]  # rotated by protein so neighbouring cards differ
    about: str  # what the line is for


CAT_LINES = (
    Line("adult", "Сухий корм для дорослих котів {p}", "Сухий корм", False, 32, 15, 9, 2.0, 589, ("brand-purina-one-cat", "brand-purina-one-dualnature"),
         "Повнораціонний корм для дорослих котів від 1 року з таурином для серця та зору."),
    Line("sterilized", "Сухий корм для стерилізованих котів {p}", "Сухий корм для стерилізованих котів", False, 35, 11, 9,
         2.0, 629, ("brand-purina-one-sterilised", "brand-brit-care-cat-sterilised", "brand-whiskas-sterile", "brand-club4paws-urinary"),
         "Знижений вміст жиру допомагає підтримувати оптимальну вагу після стерилізації, мінерали — здоров'я сечовивідних шляхів."),
    Line("kitten", "Сухий корм для кошенят {p}", "Сухий корм для кошенят", False, 36, 18, 9, 1.5, 549, ("brand-royal-canin-kitten",),
         "Для кошенят від 2 до 12 місяців: більше білка й енергії для росту, дрібні гранули для маленьких зубів."),
    Line("wet", "Вологий корм для котів {p}, пауч 85 г", "Вологий корм", True, 10, 5, 80, 0.085, 39, ("brand-felix-soup", "brand-gourmet-gold", "brand-sheba-fresh-fine", "brand-whiskas-pate", "brand-gourmet-perle", "brand-royal-canin-sterilised-pouch"),
         "Шматочки в соусі для щоденного годування або як доповнення до сухого корму; підтримує водний баланс."),
)
DOG_LINES = (
    Line("adult", "Сухий корм для дорослих собак {p}", "Сухий корм", False, 26, 14, 10, 3.0, 749, ("brand-purina-one-dog-mini", "brand-brit-lamb-buckwheat", "brand-purina-one-dog-medium", "brand-brit-lamb-rice"),
         "Повнораціонний корм для дорослих собак середніх порід з глюкозаміном для суглобів."),
    Line("puppy", "Сухий корм для цуценят {p}", "Сухий корм для цуценят", False, 30, 17, 10, 3.0, 789, ("brand-royal-canin-puppy-xsmall", "brand-royal-canin-puppy-chihuahua"),
         "Для цуценят до 12 місяців: кальцій і фосфор у правильному співвідношенні для росту кісток."),
    Line("wet", "Вологий корм для собак {p}, консерва 400 г", "Вологий корм", True, 9, 6, 78, 0.4, 109, ("dog-wet-can", "dog-wet-bowl"),
         "Ніжний паштет для щоденного годування або як смачна добавка до сухого корму."),
)


def _food(species: str, protein: Protein, line: Line, index: int) -> dict[str, Any]:
    product_id = f"PETMIND-DEMO-{species.upper()}-{line.key.upper()}-{protein.key.upper()}"
    image = line.images[index % len(line.images)]
    return {
        "id": product_id, "title": line.title.format(p=protein.ua_with), "brand": BRAND_PHOTOS.get(image, BRAND),
        "category": line.category, "species": species, "price": round(line.base_price + 20 * (index % 4), 2),
        "image_url": f"/products/{image}.jpg",
        "description": f"{line.about} {protein.note}",
        "ingredients_text": protein.wet if line.wet else protein.dry,
        "protein": float(line.protein), "fat": float(line.fat), "moisture": float(line.moisture),
        "is_regular": True, "package_weight_kg": line.weight_kg,
        "energy_kcal_per_kg": WET_KCAL if line.wet else DRY_KCAL,
    }


def _item(product_id: str, title: str, category: str, species: str, price: float, image: str, description: str,
          ingredients: str = "", *, protein: float | None = None, fat: float | None = None,
          moisture: float | None = None, weight: float | None = None, kcal: float | None = None,
          regular: bool = True) -> dict[str, Any]:
    return {"id": f"PETMIND-DEMO-{product_id}", "title": title, "brand": BRAND_PHOTOS.get(image, BRAND), "category": category,
            "species": species, "price": price, "image_url": f"/products/{image}.jpg", "description": description,
            "ingredients_text": ingredients, "protein": protein, "fat": fat, "moisture": moisture, "is_regular": regular,
            "package_weight_kg": weight, "energy_kcal_per_kg": kcal}


def _special_food() -> list[dict[str, Any]]:
    return [
        # Cats
        _item("CAT-SENIOR-SALMON", "Вологий корм для котів 7+ з лососем, пауч 85 г", "Вологий корм", "cat", 45,
              "brand-royal-canin-sterilised-loaf", "Для літніх котів: м'яка текстура, помірна калорійність і антиоксиданти для активного віку.",
              "salmon (40%), fish broth, rice, minerals, taurine", protein=10, fat=4, moisture=81, weight=0.085, kcal=WET_KCAL),
        _item("CAT-KITTEN-WET-TURKEY", "Вологий корм для кошенят з індичкою, пауч 85 г", "Вологий корм для кошенят", "cat", 42,
              "brand-royal-canin-babycat", "Ніжний мус для кошенят від 2 місяців — легко їсти навіть найменшим.",
              "turkey (45%), poultry broth, rice, minerals, taurine", protein=11, fat=6, moisture=80, weight=0.085, kcal=WET_KCAL),
        _item("CAT-GRAINFREE-TUNA", "Беззерновий сухий корм для котів з тунцем", "Сухий корм", "cat", 699, "cat-dry-grainfree",
              "Без злаків: джерело вуглеводів — горох і картопля. Підходить котам, чутливим до зернових.",
              "tuna (32%), potato, peas, fish oil, taurine", protein=34, fat=14, moisture=9, weight=2.0, kcal=DRY_KCAL),
        _item("CAT-SENSITIVE-LAMB", "Гіпоалергенний сухий корм для котів з ягнятиною", "Сухий корм", "cat", 719, "cat-dry",
              "Одне джерело тваринного білка та рис — для котів з харчовою чутливістю.",
              "lamb (30%), rice, sunflower oil, minerals, taurine", protein=31, fat=13, moisture=9, weight=2.0, kcal=DRY_KCAL),
        _item("CAT-VET-URINARY", "Лікувальний корм для котів Urinary з лососем", "Лікувальний корм", "cat", 789, "brand-royal-canin-urinary",
              "Ветеринарна дієта для підтримки здоров'я сечовивідних шляхів. Застосовуйте за рекомендацією ветеринара.",
              "salmon (25%), rice, peas, fish oil, minerals", protein=33, fat=12, moisture=8, weight=1.5, kcal=DRY_KCAL),
        _item("CAT-VET-RENAL", "Лікувальний корм для котів Satiety для контролю ваги", "Лікувальний корм", "cat", 819, "brand-royal-canin-satiety",
              "Ветеринарна дієта з високим вмістом клітковини для зниження ваги та відчуття ситості. Застосовуйте за рекомендацією ветеринара.",
              "whitefish (20%), rice, cellulose, fish oil, minerals", protein=34, fat=9, moisture=8, weight=1.5, kcal=DRY_KCAL),
        _item("CAT-VET-GASTRO", "Лікувальний корм для котів Gastro з індичкою", "Лікувальний корм", "cat", 799, "brand-royal-canin-gastro-cat",
              "Легкозасвоювана дієта для котів із чутливим травленням. Застосовуйте за рекомендацією ветеринара.",
              "turkey (25%), rice, beet pulp, fish oil, minerals", protein=32, fat=14, moisture=8, weight=1.5, kcal=DRY_KCAL),
        _item("CAT-TREATS-SALMON", "Ласощі для котів — хрусткі подушечки з лососем, 60 г", "Ласощі", "cat", 69, "brand-brit-superfruits-salmon",
              "Хрусткі подушечки з ніжною начинкою для заохочення та гри. До 10 штук на день.",
              "salmon (20%), wheat flour, rice, fish oil", weight=0.06, regular=False),
        _item("CAT-TREATS-DENTAL", "Дентальні ласощі для котів з тунцем, 60 г", "Ласощі", "cat", 74, "brand-temptations",
              "Пориста текстура механічно очищає зубний наліт під час жування.",
              "tuna (18%), rice, cellulose, minerals", weight=0.06, regular=False),
        # Dogs
        _item("DOG-LARGE-BEEF", "Сухий корм для собак великих порід з яловичиною", "Сухий корм", "dog", 1349, "dog-dry-large",
              "Для собак вагою від 25 кг: великі гранули, хондроїтин і глюкозамін для суглобів.",
              "beef (26%), rice, potato, sunflower oil, minerals", protein=25, fat=13, moisture=10, weight=12.0, kcal=DRY_KCAL),
        _item("DOG-SMALL-SALMON", "Сухий корм для собак малих порід з лососем", "Сухий корм", "dog", 499, "dog-dry",
              "Для собак до 10 кг: дрібні гранули та підвищена калорійність для активного обміну речовин.",
              "salmon (28%), rice, peas, salmon oil, minerals", protein=28, fat=17, moisture=10, weight=1.5, kcal=DRY_KCAL),
        _item("DOG-SENIOR-LAMB", "Сухий корм для собак 7+ з ягнятиною", "Сухий корм", "dog", 819, "brand-royal-canin-mini-adult-8",
              "Для літніх собак: помірна калорійність, антиоксиданти та підтримка суглобів.",
              "lamb (24%), rice, barley, flaxseed, minerals", protein=24, fat=11, moisture=10, weight=3.0, kcal=DRY_KCAL),
        _item("DOG-LIGHT-TURKEY", "Сухий корм Light для собак з надмірною вагою, з індичкою", "Сухий корм", "dog", 799, "brand-brit-weight-loss",
              "На 25% менше жиру та більше клітковини для відчуття ситості й контролю ваги.",
              "turkey (24%), rice, cellulose, peas, minerals", protein=27, fat=8, moisture=10, weight=3.0, kcal=DRY_KCAL),
        _item("DOG-SENSITIVE-SALMON", "Гіпоалергенний сухий корм для собак з лососем і картоплею", "Сухий корм", "dog", 869, "dog-dry",
              "Одне джерело тваринного білка без злаків — для собак з харчовою чутливістю.",
              "salmon (30%), potato, peas, salmon oil, minerals", protein=26, fat=14, moisture=10, weight=3.0, kcal=DRY_KCAL),
        _item("DOG-VET-GASTRO", "Лікувальний корм для собак Gastro з куркою", "Лікувальний корм", "dog", 949, "brand-royal-canin-gastro-dog",
              "Легкозасвоювана дієта для собак із розладами травлення. Застосовуйте за рекомендацією ветеринара.",
              "chicken (22%), rice, beet pulp, fish oil, minerals", protein=25, fat=12, moisture=9, weight=2.0, kcal=DRY_KCAL),
        _item("DOG-VET-RENAL", "Лікувальний корм для собак Renal з ягнятиною", "Лікувальний корм", "dog", 989, "dog-vet-diet",
              "Ветеринарна дієта зі зниженим вмістом фосфору та білка для підтримки нирок. Лише за призначенням ветеринара.",
              "lamb (15%), rice, potato, fish oil, minerals", protein=15, fat=18, moisture=9, weight=2.0, kcal=DRY_KCAL),
        _item("DOG-VET-MOBILITY", "Лікувальний корм для собак Mobility з лососем", "Лікувальний корм", "dog", 1029, "dog-vet-diet",
              "Ветеринарна дієта з омега-3 та глюкозаміном для підтримки суглобів. За рекомендацією ветеринара.",
              "salmon (24%), rice, barley, salmon oil, minerals", protein=26, fat=13, moisture=9, weight=2.0, kcal=DRY_KCAL),
        _item("DOG-WET-BOWL-BEEF", "Вологий корм для собак — рагу з яловичиною та овочами, 300 г", "Вологий корм", "dog", 89,
              "dog-wet-bowl", "Шматочки м'яса з морквою та горошком у ніжному соусі.",
              "beef (40%), carrots, peas, beef broth, minerals", protein=9, fat=5, moisture=79, weight=0.3, kcal=WET_KCAL),
        _item("DOG-TREATS-BISCUITS", "Печиво для собак «Кісточки» з яловичиною, 400 г", "Ласощі", "dog", 119, "dog-biscuits",
              "Хрустке печиво для дресирування та заохочення. Годуйте як доповнення до основного раціону.",
              "wheat flour, beef (10%), oats, sunflower oil", weight=0.4, regular=False),
        _item("DOG-TREATS-MINI", "Міні-кісточки для собак малих порід з ягнятиною, 200 г", "Ласощі", "dog", 79, "dog-biscuits-2",
              "Маленьке печиво для дрібних порід і цуценят від 3 місяців.",
              "wheat flour, lamb (8%), rice, sunflower oil", weight=0.2, regular=False),
        _item("DOG-DENTAL-STICKS", "Дентальні палички для собак середніх порід, 7 шт.", "Ласощі", "dog", 149, "brand-brit-dental-stick",
              "Щоденна жувальна паличка зменшує утворення зубного нальоту та освіжає дихання.",
              "rice flour, cellulose, potato starch, minerals", weight=0.18, regular=False),
        # Rabbits
        _item("RABBIT-PELLETS", "Гранульований корм для кролів з тимофіївкою, 1,5 кг", "Корм для кролів", "rabbit", 289,
              "rabbit-pellets", "Повнораціонні гранули на основі сіна з високим вмістом клітковини для здорового травлення та стирання зубів.",
              "timothy hay, alfalfa, wheat bran, minerals, vitamins", protein=14, fat=3, moisture=11, weight=1.5),
        _item("RABBIT-HAY", "Лугове сіно для кролів і гризунів, 1 кг", "Корм для кролів", "rabbit", 149, "rabbit-hay",
              "Ароматне сіно першого укосу — основа раціону кролів, морських свинок і шиншил.",
              "meadow hay", weight=1.0),
        # Rodents
        _item("RODENT-GUINEA-PELLETS", "Корм для морських свинок з вітаміном C, 1 кг", "Корм для гризунів", "rodent", 239,
              "guinea-pellets", "Гранули з тимофіївкою та стабілізованим вітаміном C, який морські свинки не синтезують самостійно.",
              "timothy hay, alfalfa, minerals, vitamins", protein=15, fat=3, moisture=11, weight=1.0),
        _item("RODENT-HAMSTER-MIX", "Зернова суміш для хом'яків і мишей, 500 г", "Корм для гризунів", "rodent", 119, "hamster-mix",
              "Суміш зерен, насіння та сушених овочів для щоденного годування декоративних гризунів.",
              "oats, millet, sunflower seeds, corn, carrots", protein=13, fat=6, moisture=10, weight=0.5),
        _item("RODENT-TREATS-VEG", "Ласощі для гризунів — овочеві палички, 2 шт.", "Корм для гризунів", "rodent", 69, "guinea-treats",
              "Палички з морквою та петрушкою для гризіння й стирання зубів.",
              "wheat flour, carrots, parsley, honey", weight=0.1, regular=False),
        # Birds
        _item("BIRD-BUDGIE-MIX", "Корм для хвилястих папуг, 1 кг", "Корм для птахів", "bird", 129, "bird-seed",
              "Збалансована суміш проса, вівса та насіння канаркової трави для щоденного годування.",
              "millet, oats, canary seed, minerals", protein=12, fat=5, moisture=11, weight=1.0),
        _item("BIRD-PARROT-MIX", "Корм для середніх і великих папуг, 1 кг", "Корм для птахів", "bird", 219, "bird-seed",
              "Суміш насіння соняшника, кукурудзи, проса й сушених фруктів для корел, нерозлучників і жако.",
              "sunflower seeds, corn, millet, oats, dried apple", protein=14, fat=15, moisture=10, weight=1.0),
        # Fish
        _item("FISH-FLAKES", "Корм-пластівці для акваріумних риб, 250 мл", "Корм для риб", "fish", 159, "fish-flakes",
              "Пластівці для всіх тропічних риб, що годуються біля поверхні; не каламутять воду.",
              "fish meal, wheat flour, shrimp, spirulina, vitamins", protein=46, fat=8, moisture=7, weight=0.05),
        _item("FISH-GOLDFISH", "Корм для золотих рибок і коропових, гранули 250 мл", "Корм для риб", "fish", 149, "fish-flakes",
              "Плаваючі гранули з рослинними компонентами для золотих рибок і кої.",
              "fish meal, wheat flour, spirulina, minerals", protein=40, fat=6, moisture=8, weight=0.06),
    ]


def _pharmacy() -> list[dict[str, Any]]:
    fleas = "Ветаптека · Від бліх і кліщів"
    worms = "Ветаптека · Від гельмінтів"
    vit = "Ветаптека · Вітаміни та добавки"
    care = "Ветаптека · Гігієна та догляд"
    return [
        _item("VET-SPOTON-CAT", "Краплі на холку від бліх і кліщів для котів 1–4 кг, 3 піпетки", fleas, "cat", 289, "vet-pipettes",
              f"Захищають від бліх і кліщів до 4 тижнів після нанесення. Діюча речовина — фіпроніл. {VET_NOTE}", "fipronil"),
        _item("VET-SPOTON-DOG-M", "Краплі на холку від бліх і кліщів для собак 10–20 кг, 3 піпетки", fleas, "dog", 369, "vet-drops",
              f"Захист від бліх і кліщів до 4 тижнів. Діюча речовина — фіпроніл. Лише для собак. {VET_NOTE}", "fipronil"),
        _item("VET-SPOTON-DOG-L", "Краплі на холку від бліх і кліщів для собак 20–40 кг, 3 піпетки", fleas, "dog", 449, "vet-pipettes",
              f"Захист від бліх і кліщів до 4 тижнів для великих собак. Діюча речовина — фіпроніл. {VET_NOTE}", "fipronil"),
        _item("VET-COLLAR-CAT", "Нашийник від бліх і кліщів для котів, 38 см", fleas, "cat", 329, "vet-collar",
              f"Еластичний нашийник з безпечною застібкою, захищає до 8 місяців. Діючі речовини — імідаклоприд і флуметрин. {VET_NOTE}",
              "imidacloprid, flumethrin"),
        _item("VET-COLLAR-DOG", "Нашийник від бліх і кліщів для собак, 70 см", fleas, "dog", 489, "vet-collar",
              f"Водостійкий нашийник для собак, захищає до 8 місяців. Діючі речовини — імідаклоприд і флуметрин. {VET_NOTE}",
              "imidacloprid, flumethrin"),
        _item("VET-TABS-DOG", "Жувальні таблетки від бліх і кліщів для собак 10–25 кг, 1 шт.", fleas, "dog", 579, "vet-tablets",
              f"Смачна жувальна таблетка діє до 12 тижнів. Препарат групи ізоксазолінів; відпускається після консультації ветеринара. {VET_NOTE}",
              "isoxazoline, flavouring"),
        _item("VET-SHAMPOO-FLEA", "Шампунь від бліх для собак, 250 мл", fleas, "dog", 189, "shampoo-flea",
              f"Допомагає позбутися бліх під час купання та заспокоює подразнену шкіру. Лише для зовнішнього застосування. {VET_NOTE}",
              "fipronil, aloe vera"),
        _item("VET-DEWORM-CAT", "Таблетки від глистів для котів, 2 шт.", worms, "cat", 149, "vet-tablets",
              f"Діють проти круглих і стрічкових гельмінтів. Діючі речовини — празиквантел і пірантелу памоат. {VET_NOTE}",
              "praziquantel, pyrantel pamoate"),
        _item("VET-DEWORM-DOG", "Таблетки від глистів для собак, 6 шт.", worms, "dog", 239, "vet-tablets",
              f"Комплексна дегельмінтизація собак. Діючі речовини — празиквантел, пірантелу памоат, фебантел. {VET_NOTE}",
              "praziquantel, pyrantel pamoate, febantel"),
        _item("VET-SALMON-OIL-CAT", "Олія лосося для котів, 250 мл", vit, "cat", 259, "salmon-oil",
              "Натуральна олія лосося з омега-3 та омега-6 для блискучої шерсті й здорової шкіри. Додавайте до корму щодня.",
              "salmon oil", weight=0.25),
        _item("VET-SALMON-OIL-DOG", "Олія лосося для собак, 500 мл", vit, "dog", 389, "vet-omega-bottle",
              "Олія лосося холодного віджиму з омега-3 для шерсті, шкіри та суглобів. Зручний дозатор.",
              "salmon oil", weight=0.5),
        _item("VET-FISH-OIL-CAPS", "Риб'ячий жир у капсулах для собак і котів, 60 шт.", vit, "dog", 299, "fish-oil",
              "Капсули з омега-3 жирними кислотами (EPA та DHA) для підтримки шкіри, шерсті й серця.",
              "fish oil, gelatin", weight=0.08),
        _item("VET-MULTIVIT-CAT", "Мультивітаміни для котів з таурином, 100 таблеток", vit, "cat", 199, "vet-cat-checkup",
              "Вітамінно-мінеральний комплекс для щоденної підтримки імунітету, зору та серця.",
              "brewers yeast, taurine, vitamins, minerals"),
        _item("VET-MULTIVIT-DOG", "Мультивітаміни для собак, 100 таблеток", vit, "dog", 229, "vet-vitamins",
              "Щоденний вітамінно-мінеральний комплекс для собак усіх порід.",
              "brewers yeast, vitamins, minerals"),
        _item("VET-JOINT-DOG", "Добавка для суглобів собак з глюкозаміном, 60 таблеток", vit, "dog", 419, "vet-vitamins",
              "Глюкозамін, хондроїтин і MSM для підтримки рухливості суглобів у літніх і великих собак.",
              "glucosamine, chondroitin, msm"),
        _item("VET-MALT-PASTE-CAT", "Паста для виведення шерсті для котів, 100 г", vit, "cat", 169, "cat-grooming",
              "Мальт-паста допомагає виводити грудочки шерсті зі шлунка, особливо в період линьки.",
              "malt extract, vegetable oils, vitamins"),
        _item("VET-PROBIOTIC", "Пробіотик для котів і собак, 30 саше", vit, "dog", 349, "vet-capsules",
              "Корисні бактерії для підтримки мікрофлори кишечника, зокрема після зміни корму.",
              "probiotic cultures, maltodextrin"),
        _item("VET-EAR-CLEANER", "Лосьйон для очищення вух котів і собак, 100 мл", care, "dog", 179, "vet-dog-ear",
              "М'яко розчиняє вушну сірку та забруднення. Для регулярної гігієни вух.",
              "chlorhexidine, aloe vera"),
        _item("VET-EYE-LOTION", "Лосьйон для догляду за очима котів і собак, 50 мл", care, "cat", 129, "vet-cat-ear",
              "Делікатно очищає ділянку навколо очей від виділень і слідів сліз.",
              "chamomile extract, boric acid"),
        _item("VET-SHAMPOO-CAT", "Делікатний шампунь для котів, 250 мл", care, "cat", 169, "shampoo-gentle",
              "М'яка формула з алое вера для чутливої шкіри; не пересушує шерсть.",
              "aloe vera, coconut surfactants"),
        _item("VET-SHAMPOO-DOG", "Шампунь для собак з алое вера, 500 мл", care, "dog", 199, "shampoo-gentle",
              "Для регулярного купання: очищає, зволожує шкіру та полегшує розчісування.",
              "aloe vera, coconut surfactants"),
        _item("VET-RABBIT-VIT", "Вітаміни для кролів і гризунів у воду, 50 мл", vit, "rabbit", 119, "vet-drops",
              "Рідкий комплекс вітамінів A, D3, E та C — додається в питну воду.",
              "vitamins"),
        _item("VET-RODENT-SPRAY", "Спрей від паразитів для кролів і гризунів, 100 мл", fleas, "rodent", 189, "vet-spray",
              f"Для обробки шерсті та клітки від бліх, вошей і кліщів. {VET_NOTE}", "pyrethrins"),
        _item("VET-BIRD-VIT", "Вітаміни для птахів у воду, 15 мл", vit, "bird", 99, "bird-vitamins",
              "Вітамінний комплекс для хвилястих папуг, канарок і корел, особливо в період линьки.",
              "vitamins, amino acids"),
        _item("VET-FISH-CONDITIONER", "Кондиціонер для акваріумної води, 250 мл", care, "fish", 159, "vet-drops",
              "Знешкоджує хлор і важкі метали у водопровідній воді та захищає слизову оболонку риб.",
              "sodium thiosulfate, aloe vera"),
    ]


def demo_catalog() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for species, proteins, lines in (("cat", CAT_PROTEINS, CAT_LINES), ("dog", DOG_PROTEINS, DOG_LINES)):
        for p_index, protein in enumerate(proteins):
            for line in lines:
                rows.append(_food(species, protein, line, p_index))
    return rows + _special_food() + _pharmacy()
