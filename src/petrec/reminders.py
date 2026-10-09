"""Read-only reminder formatting for API responses."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import math
from typing import Any, Iterable, Literal, Mapping

from pydantic import BaseModel, Field

from .llm import LLMClient

ReminderSeverity = Literal["скоро", "сьогодні", "прострочено"]

# Starting-point activity multipliers. They are configurable because individual
# calorie needs vary substantially; they are not a veterinary prescription.
ENERGY_FACTORS = {
    "cat": {"sterilized": 1.2, "sterilised": 1.2, "indoor": 1.2, "default": 1.4},
    "dog": {"sterilized": 1.6, "sterilised": 1.6, "active": 1.8, "default": 1.4},
}
CATEGORY_RULES: dict[str, tuple[int, float]] = {
    "dry_food": (5, 0.9), "wet_food": (4, 0.8), "parasite": (3, 1.0),
    "deworming": (7, 1.0), "litter": (4, 0.6), "supplement": (5, 0.7),
    "default": (5, 0.7),
}


class ReminderItem(BaseModel):
    """A purchase reminder rendered for the Petly client."""

    item_id: str
    title: str
    days_left: int
    expected_date: date
    text: str
    severity: ReminderSeverity


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _template(title: str, days_left: int) -> str:
    if days_left < 0:
        return f"Час поповнити запас «{title}»: рекомендована дата покупки минула на {-days_left} дн."
    if days_left == 0:
        return f"Сьогодні час поповнити запас «{title}»."
    return f"Незабаром знадобиться «{title}» — орієнтовно через {days_left} дн."


def expected_restock_date(purchase_dates: Iterable[date | datetime]) -> date | None:
    """Forecast the next replenishment date from the mean positive repeat interval."""
    dates = sorted(_as_date(value) for value in purchase_dates)
    if len(dates) < 2:
        return None
    intervals = [(right - left).days for left, right in zip(dates, dates[1:])]
    intervals = [interval for interval in intervals if interval > 0]
    if not intervals:
        return None
    average_interval = max(1, round(sum(intervals) / len(intervals)))
    return dates[-1] + timedelta(days=average_interval)


def smoothed_purchase_interval(purchase_dates: Iterable[date | datetime], alpha: float = 0.3) -> float | None:
    """Forecast the next interval with simple exponential smoothing (KRM 3.5–3.6)."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    dates = sorted(_as_date(value) for value in purchase_dates)
    intervals = [(right - left).days for left, right in zip(dates, dates[1:])]
    intervals = [float(value) for value in intervals if value > 0]
    if not intervals:
        return None
    forecast = intervals[0]
    for actual in intervals[1:]:
        forecast = alpha * actual + (1 - alpha) * forecast
    return forecast


def estimate_food_stock_days(
    weight_kg: float, species: str, energy_kcal_per_kg: float, package_weight_kg: float,
    quantity: int = 1, lifestyle: str | None = None, energy_factor: float | None = None,
) -> float:
    """Estimate package duration from RER, lifestyle factor and food energy (KRM 3.7–3.10)."""
    values = (weight_kg, energy_kcal_per_kg, package_weight_kg)
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in values) or quantity < 1:
        raise ValueError("weight, food energy, package weight and quantity must be positive")
    species = species.lower()
    if species not in ENERGY_FACTORS:
        raise ValueError("species must be 'cat' or 'dog'")
    lifestyle_key = (lifestyle or "").strip().lower()
    factor = energy_factor if energy_factor is not None else ENERGY_FACTORS[species].get(
        lifestyle_key, ENERGY_FACTORS[species]["default"]
    )
    if not math.isfinite(float(factor)) or factor <= 0:
        raise ValueError("energy_factor must be positive")
    rer = 70.0 * float(weight_kg) ** 0.75
    daily_kcal = rer * factor
    daily_grams = daily_kcal / float(energy_kcal_per_kg) * 1000.0
    return float(package_weight_kg) * quantity * 1000.0 / daily_grams


def predict_purchase_date(
    purchase_dates: Iterable[date | datetime], *, stock_days: float | None,
    alpha: float = 0.3, stock_weight: float | None = None,
) -> date | None:
    """Blend observed repeat interval and estimated stock duration (KRM 3.11)."""
    dates = sorted(_as_date(value) for value in purchase_dates)
    if not dates:
        return None
    if stock_weight is None:
        stock_weight = 1.0 if len(dates) < 3 else 0.5
    if not 0 <= stock_weight <= 1:
        raise ValueError("stock_weight must be in [0, 1]")
    observed = smoothed_purchase_interval(dates, alpha)
    if stock_days is None and observed is None:
        return None
    if stock_days is None:
        interval = observed
    elif observed is None or stock_weight == 1:
        interval = stock_days
    else:
        interval = stock_weight * stock_days + (1 - stock_weight) * observed
    return dates[-1] + timedelta(days=max(1, round(float(interval))))


def reminder_priority(expected_date: date, today: date, category_importance: float, scale_days: float = 30.0) -> float:
    """Rank reminders by category importance and due-date proximity (KRM 3.13)."""
    if not 0 <= category_importance <= 1 or scale_days <= 0:
        raise ValueError("importance must be in [0, 1] and scale_days must be positive")
    remaining = max(0, (expected_date - today).days)
    return float(category_importance / (1 + remaining / scale_days))


def get_reminders(
    user_id: Any,
    pet_id: Any,
    today: date | datetime | str,
    *,
    items: Iterable[Mapping[str, Any]] = (),
    llm_client: LLMClient | None = None,
) -> list[ReminderItem]:
    """Build reminders from caller-provided due-item records without mutating state.

    Each input mapping must include ``item_id``, ``title`` and ``expected_date``.
    This function performs no persistence, scheduling, notification, or purchase
    operation; the API can supply read-only due records from its data layer.
    LLM wording is optional and falls back to a deterministic Ukrainian template.
    ``user_id`` and ``pet_id`` are accepted for API symmetry and source-side lookup.
    """
    del user_id, pet_id  # identity filtering belongs to the API's read-only data query
    current_date = _as_date(today)
    reminders: list[ReminderItem] = []
    for item in items:
        expected_date = _as_date(item["expected_date"])
        days_left = (expected_date - current_date).days
        severity: ReminderSeverity = "прострочено" if days_left < 0 else "сьогодні" if days_left == 0 else "скоро"
        item_id = str(item["item_id"])
        title = str(item["title"])
        text = _template(title, days_left)
        if llm_client is not None:
            system = ("Сформулюй коротке дружнє нагадування українською. Не додавай фактів, "
                      "дат чи порад, яких немає у вхідних даних. Поверни лише текст до 200 символів.")
            user = (f"Товар: {title}\nОчікувана дата: {expected_date.isoformat()}\n"
                    f"Днів до дати: {days_left}\nТерміновість: {severity}")
            try:
                generated = llm_client.complete(system, user).strip()
                if generated:
                    text = generated[:200]
            except Exception:
                # An unavailable model must not prevent the API from returning reminders.
                pass
        reminders.append(ReminderItem(
            item_id=item_id,
            title=title,
            days_left=days_left,
            expected_date=expected_date,
            text=text,
            severity=severity,
        ))
    return reminders
