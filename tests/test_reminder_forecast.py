from datetime import date, timedelta

import pytest

from petrec.reminders import (
    estimate_food_stock_days,
    predict_purchase_date,
    reminder_priority,
    smoothed_purchase_interval,
)


def test_cat_food_stock_example_matches_koren_thesis_example():
    days = estimate_food_stock_days(
        weight_kg=4.5,
        species="cat",
        energy_kcal_per_kg=3800,
        package_weight_kg=2,
        lifestyle="sterilized",
    )
    assert days == pytest.approx(29.3, abs=0.2)


def test_interval_uses_simple_exponential_smoothing():
    dates = [date(2026, 1, 1), date(2026, 1, 11), date(2026, 1, 26)]
    assert smoothed_purchase_interval(dates, alpha=0.5) == pytest.approx(12.5)


def test_fewer_than_three_purchases_use_stock_estimate_only():
    dates = [date(2026, 1, 1), date(2026, 1, 11)]
    assert predict_purchase_date(dates, stock_days=20) == date(2026, 1, 31)


def test_three_or_more_purchases_blend_history_and_stock():
    dates = [date(2026, 1, 1), date(2026, 1, 11), date(2026, 1, 26)]
    # α=.3 gives 12.5 days; with ω=.5, expected interval is 16.25 -> 16 days.
    assert predict_purchase_date(dates, stock_days=20, alpha=0.5) == date(2026, 2, 11)


def test_priority_increases_as_restock_date_gets_closer():
    today = date(2026, 1, 1)
    soon = reminder_priority(today + timedelta(days=2), today, 0.9)
    later = reminder_priority(today + timedelta(days=20), today, 0.9)
    assert soon > later
