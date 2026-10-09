"""Deterministic preference updates derived from explicit user reactions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def update_preference_state(
    state: Mapping[str, Any] | None,
    event_type: str,
    product_id: str,
    *,
    category: str | None = None,
) -> dict[str, Any]:
    """Return a new preference-state value; the caller is responsible for persistence."""
    if event_type not in {"view", "add_to_cart", "dismiss", "like", "purchase"}:
        raise ValueError(f"Unsupported preference event: {event_type}")
    result = dict(state or {})
    events = dict(result.get("events", {}))
    events[event_type] = int(events.get(event_type, 0)) + 1
    result["events"] = events
    key = "disliked_products" if event_type == "dismiss" else "liked_products" if event_type in {"like", "purchase"} else "interested_products"
    product_ids = list(result.get(key, []))
    if product_id not in product_ids:
        product_ids.append(product_id)
    result[key] = product_ids[-100:]
    if category and event_type in {"like", "purchase", "dismiss"}:
        category_key = "disliked_categories" if event_type == "dismiss" else "preferred_categories"
        categories = list(result.get(category_key, []))
        if category not in categories:
            categories.append(category)
        result[category_key] = categories[-50:]
    return result
