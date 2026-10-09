from datetime import date

from petrec.llm import MockLLM
from petrec.reminders import get_reminders


def test_get_reminders_uses_llm_and_calculates_severity():
    llm = MockLLM("Незабаром перевірте запас корму для улюбленця.")
    reminders = get_reminders(
        "u1", "p1", date(2026, 10, 5),
        items=[{"item_id": "food-1", "title": "Корм", "expected_date": "2026-10-08"}],
        llm_client=llm,
    )
    assert len(reminders) == 1
    assert reminders[0].days_left == 3
    assert reminders[0].severity == "скоро"
    assert reminders[0].text.startswith("Незабаром")


def test_get_reminders_falls_back_to_template_for_due_and_overdue():
    reminders = get_reminders("u1", "p1", date(2026, 10, 5), items=[
        {"item_id": "today", "title": "Вологий корм", "expected_date": "2026-10-05"},
        {"item_id": "late", "title": "Наповнювач", "expected_date": "2026-10-02"},
    ])
    assert [item.severity for item in reminders] == ["сьогодні", "прострочено"]
    assert reminders[0].days_left == 0
    assert reminders[1].days_left == -3
    assert "Сьогодні" in reminders[0].text
    assert "поповнити запас" in reminders[1].text


def test_get_reminders_is_empty_without_due_records():
    assert get_reminders("u1", "p1", date(2026, 10, 5)) == []
