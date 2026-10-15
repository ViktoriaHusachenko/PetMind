"""System prompt of the PetMind assistant."""

from __future__ import annotations

from typing import Any

SPECIES_UA = {"cat": "кіт", "dog": "собака", "rabbit": "кролик", "bird": "птах", "rodent": "гризун",
              "reptile": "рептилія", "fish": "риба", "other": "інша тварина"}

SYSTEM_PROMPT = """Ти — ШІ-помічник зоомагазину PetMind. Відповідай українською, коротко й по суті.
Факти про профіль тварини, товари, склад, нагадування й рекомендації бери лише з інструментів.
Не вигадуй товарів, цін чи складу. Рішення про безпечність приймає лише інструмент check_safety:
ніколи не радь товар, для якого safe=false, і не стверджуй, що товар «без алергенів», якщо його не перевірено.
Перш ніж радити конкретний товар, знайдений через search_products, перевір його через check_safety.
Не показуй ідентифікаторів товарів користувачу — назви товарів достатньо, картки покаже інтерфейс.
Не давай медичних порад, діагнозів чи дозувань ліків — порадь звернутися до ветеринара.
Якщо запит поза темою зоомагазину, ввічливо поверни розмову до потреб тварини.
Оформлення: пиши звичайним текстом без Markdown — без зірочок, решіток і підкреслень. Перелік товарів
подавай окремими рядками, кожен починай із «• », не більше 5 пунктів. Назву товару бери в «лапки».
Не починай речення зі службових символів."""


def pet_summary(profile: dict[str, Any]) -> str:
    """Pet facts for the prompt; owner personal data is never included."""
    parts = [f"Ім'я: {profile.get('name') or 'невідомо'}"]
    if profile.get("species"):
        parts.append(f"вид: {SPECIES_UA.get(profile['species'], profile['species'])}")
    if profile.get("age_group"):
        parts.append(f"вік: {profile['age_group']}")
    if profile.get("weight_kg"):
        parts.append(f"вага: {profile['weight_kg']} кг")
    allergens = profile.get("allergens") or []
    parts.append("алергени: " + (", ".join(map(str, allergens)) if allergens else "не вказані"))
    if profile.get("lifestyle"):
        parts.append(f"спосіб життя: {profile['lifestyle']}")
    if profile.get("diet_restrictions"):
        parts.append("дієтичні обмеження: " + ", ".join(map(str, profile["diet_restrictions"])))
    return "; ".join(parts)


def build_system_prompt(profile: dict[str, Any], pet_id: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nПоточна тварина (pet_id={pet_id}): {pet_summary(profile)}."
