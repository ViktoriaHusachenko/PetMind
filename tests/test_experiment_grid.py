from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from run_all import _grid_table, _options_from_row, _reminder_metrics


def test_thesis_parameter_grid_has_valid_weight_simplex():
    grid = _grid_table()
    assert len(grid) >= 90
    assert (grid[["gamma1", "gamma2", "gamma3"]].sum(axis=1).round(8) == 1).all()
    options = _options_from_row(grid.iloc[0])
    assert options.gamma1 + options.gamma2 + options.gamma3 == 1


def test_reminder_metrics_compare_baseline_and_smoothing_without_fabricating_stock():
    interactions = pd.DataFrame({
        "user_id": ["u", "u", "u", "u"],
        "parent_asin": ["p"] * 4,
        "timestamp": ["2026-01-01", "2026-01-11", "2026-01-26", "2026-02-10"],
    })
    products = pd.DataFrame({"parent_asin": ["p"], "category": ["dry_food"]})
    result = _reminder_metrics(interactions, products, None)
    assert set(result.model) == {"середній інтервал", "експоненційне згладжування"}
    assert (result.n > 0).all()
    assert (result.within_5_days.between(0, 1)).all()
