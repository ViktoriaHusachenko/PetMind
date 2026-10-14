"""Deterministic guards around the LLM: medical intent, product mentions, final cards."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .context import ToolContext, is_sterilized_product

VET_REPLY = "Я не надаю порад щодо лікування, діагнозу чи ліків. Будь ласка, зверніться до ветеринара."
URGENT_VET_REPLY = ("Це може бути небезпечно для здоров'я тварини. Будь ласка, якнайшвидше зверніться до ветеринара "
                    "або у цілодобову ветклініку. Я не надаю медичних порад.")

# Store goods whose names contain medical-sounding words; removed before intent matching.
_PRODUCT_PHRASES = re.compile(
    r"лікувальн\w*\s+(?:корм|дієт|раціон|консерв|паштет)\w*|ветеринарн\w*\s+(?:дієт|корм)\w*|"
    r"дієтичн\w*\s+корм\w*|гіпоалерген\w*|(?:renal|urinary|hypoallergenic|gastrointestinal)\s+\w+|"
    r"(?:cat|dog|dental|training)\s+treats?|\btreats\b|ласощ\w*",
    re.IGNORECASE)

_URGENT = re.compile(
    r"кров\w*|судом\w*|не\s+дихає|задиха\w*|отру\w*|непритомн\w*|"
    r"з'?ї(?:в|ла)\s+(?:шоколад|ліки|таблетк|отрут|виноград|цибул)\w*|"
    r"не\s+(?:їсть|п'?є)\s+(?:вже\s+)?(?:\d+|два|три|кілька|декілька)\s+(?:дн|доб)\w*|"
    r"температур\w*\s+(?:3[9]|4\d)|\bblood\b|seizure\w*|poison\w*|not\s+breathing",
    re.IGNORECASE)

_MEDICAL = re.compile(
    r"\bлікув(?:ати|ання|ити|ую|ує|ують)\b|\bвиліку\w*|\bлікуванн\w*|діагноз\w*|симптом\w*|"
    r"медикамент\w*|\bліки\b|\bліків\b|\bліками\b|таблетк\w*|антибіотик\w*|ін'?єкці\w*|уколи?\b|"
    r"дозуванн\w*|\bдоз[аиуою]\b|"
    r"блюв\w*|нуд(?:от|ить|ять)\w*|пронос\w*|діаре\w*|запор\w*|кашл\w*|чха\w*|кульга\w*|температур\w*|"
    r"(?:болить|біль|болі)\b|хвор\w*|запаленн\w*|інфекц\w*|"
    r"\btreat(?:ing|ment)?\b|diagnos\w*|medicin\w*|medicat\w*|\bcure\b|symptom\w*|vomit\w*|diarrh\w*|"
    r"\bdos(?:e|age)\b|antibiotic\w*|\bpills?\b|\bsick\b|\bill\b",
    re.IGNORECASE)


@dataclass(frozen=True)
class MedicalVerdict:
    blocked: bool
    urgent: bool = False

    @property
    def reply(self) -> str:
        return URGENT_VET_REPLY if self.urgent else VET_REPLY


class MedicalGuard:
    """Blocks requests for treatment, diagnosis or medication; allows therapeutic *food* shopping."""

    def check(self, text: str) -> MedicalVerdict:
        normalized = (text or "").translate(str.maketrans({"’": "'", "ʼ": "'"}))
        if _URGENT.search(normalized):
            return MedicalVerdict(True, urgent=True)
        stripped = _PRODUCT_PHRASES.sub(" ", normalized)
        return MedicalVerdict(bool(_MEDICAL.search(stripped)))


_PRODUCT_ID = re.compile(r"\b[A-Z0-9][A-Z0-9-]{7,63}\b")
# Shapes of real catalog ids (ASIN-like research data, seeded demo goods).
_CATALOG_ID_SHAPE = re.compile(r"^(?:[A-Z0-9]{8,16}|PETMIND-[A-Z0-9-]+)$")


class ProductMentionGuard:
    """The reply may name only products that check_safety approved for this pet."""

    def violations(self, reply: str, ctx: ToolContext) -> set[str]:
        candidates = {pid for pid in _PRODUCT_ID.findall(reply) if any(ch.isdigit() for ch in pid) or pid.startswith("PETMIND-")}
        folded = reply.casefold()
        for pid, basic in ctx.seen.items():
            title = str(basic.get("title") or "")
            if title and title.casefold() in folded:
                candidates.add(pid)
        bad = set()
        for product_id in candidates:
            basic = ctx.service.get_product_basic(product_id)
            if basic is None:
                # An invented id that looks like a catalog id means the model made a product up.
                if product_id in ctx.seen or _CATALOG_ID_SHAPE.match(product_id):
                    bad.add(product_id)
                continue
            if ctx.service.check_safety(product_id, ctx.pet_id).get("safe") is not True:
                bad.add(product_id)
        return bad


class FinalCardsGuard:
    """Re-verifies every card right before the response: species and the deterministic Safe."""

    def filter(self, ctx: ToolContext) -> list[dict[str, Any]]:
        output = []
        for product_id, row in list(ctx.approved.items()):
            basic = ctx.service.get_product_basic(product_id)
            if basic is None or basic.get("species") != ctx.species:
                continue
            if ctx.service.check_safety(product_id, ctx.pet_id).get("safe") is True:
                output.append({**row, "safe_badge": True, "_sterilized": is_sterilized_product(basic)})
        if ctx.sterilized and any(item["_sterilized"] for item in output):
            # Prefer sterilized-specific food when the catalog has it.
            output = [item for item in output if item["_sterilized"]]
        for item in output:
            item.pop("_sterilized", None)
        return output
