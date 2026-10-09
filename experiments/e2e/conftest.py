"""Collect live acceptance outcomes for the KRM 3.5 result table."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results" / "table_3_5.csv"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" or report.failed or report.skipped:
        item._table_3_5_result = report


def pytest_sessionfinish(session, exitstatus):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    reports = getattr(session, "_reports", {})
    for item in session.items:
        report = getattr(item, "_table_3_5_result", None)
        if report is None:
            result, evidence, duration = "SKIP", "No call phase completed", 0.0
        elif report.passed:
            result, evidence = "PASS", "HTTP assertions passed"
            duration = report.duration
        elif report.skipped:
            result, evidence = "SKIP", str(report.longreprtext).replace("\n", " ")[:1000]
            duration = report.duration
        else:
            result = "FAIL"
            evidence = str(report.longreprtext).replace("\n", " ")[:1000]
            duration = report.duration
        rows.append({"case": item.name, "result": result, "evidence": evidence,
                     "duration_seconds": f"{duration:.3f}"})
    with RESULTS.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "result", "evidence", "duration_seconds"])
        writer.writeheader()
        writer.writerows(rows)
