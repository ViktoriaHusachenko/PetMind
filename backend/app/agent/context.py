"""Per-conversation state shared by agent tools and guards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

STERILIZED_TERMS = ("steriliz", "neuter", "spay", "стерил", "кастр")
STERILIZED_NEED_TERMS = (*STERILIZED_TERMS[:2], "стерил", "спер", "кастр")


class AgentBackend(Protocol):
    """The slice of APIService the agent depends on (interface segregation)."""

    def get_pet_profile(self, user_id: str, pet_id: str) -> dict[str, Any]: ...
    def get_product_basic(self, product_id: str) -> dict[str, Any] | None: ...
    def check_safety(self, product_id: str, pet_id: str) -> dict[str, Any]: ...
    def recommend(self, user_id: str, pet_id: str, *, k: int = 10, use_llm: bool = True) -> list[dict[str, Any]]: ...
    def search_catalog(self, query: str, *, species: str | None, limit: int = 10) -> list[dict[str, Any]]: ...
    def list_reminders(self, user_id: str, pet_id: str) -> list[dict[str, Any]]: ...
    def get_product(self, product_id: str, *, pet_id: str | None = None) -> dict[str, Any] | None: ...


def mentions_sterilization(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in STERILIZED_NEED_TERMS)


def is_sterilized_product(basic: dict[str, Any]) -> bool:
    label = " ".join(str(basic.get(key) or "") for key in ("title", "category")).lower()
    return any(term in label for term in STERILIZED_TERMS)


@dataclass
class ToolContext:
    service: AgentBackend
    user_id: str
    pet_id: str
    profile: dict[str, Any]
    # Products whose safety check_safety confirmed in this conversation → card payload.
    approved: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Every product id the tools exposed to the model (used to police the final reply).
    seen: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def species(self) -> str | None:
        return self.profile.get("species")

    @property
    def sterilized(self) -> bool:
        needs = " ".join((str(self.profile.get("lifestyle") or ""), str(self.profile.get("diet_restrictions") or "")))
        return mentions_sterilization(needs)

    def approve_safe(self, product_id: str, card: dict[str, Any]) -> bool:
        """Approve a card only after the deterministic check; returns the decision."""
        decision = self.service.check_safety(product_id, self.pet_id)
        if decision.get("safe") is True:
            self.approved[product_id] = {**card, "item_id": product_id, "safe_badge": True}
            return True
        self.approved.pop(product_id, None)
        return False
