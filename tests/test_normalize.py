"""Tests against the ACTUAL supplied files.

These exist because two properties of that data are easy to get wrong and
impossible to notice once wrong: module day ranges, and the difference between
a mission that is absent and a mission that was skipped.
"""

from __future__ import annotations

import pytest

from app.normalize import (
    MissionState,
    build_profile,
    load_candidates,
    load_curriculum,
)

curriculum = load_curriculum()
candidates = load_candidates()


def by_id(cid: str):
    for c in candidates:
        if c.member.id == cid:
            return c
    raise AssertionError(f"{cid} missing from candidates.json")


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


def test_thirty_one_days_and_eight_modules():
    assert len(curriculum.days) == 31
    assert len(curriculum.modules) == 8
    assert curriculum.day_numbers == list(range(1, 32))


def test_module_days_are_ranges_not_lists():
    """`[7, 10]` means 7,8,9,10. Reading it as membership loses days 8 and 9."""
    m3 = next(m for m in curriculum.modules if m.n == 3)
    assert m3.title == "Embeddings & Vector Search"
    assert m3.days == [7, 8, 9, 10]
    assert m3.contains(9)

    # Every day in 1..31 must belong to exactly one module.
    for day in range(1, 32):
        owning = [m for m in curriculum.modules if m.contains(day)]
        assert len(owning) == 1, f"day {day} belongs to {len(owning)} modules"


def test_every_day_carries_objectives_and_a_type():
    for n, day in curriculum.days.items():
        assert day.objectives, f"day {n} has no objectives"
        assert day.title
        assert day.type
        assert day.module_n > 0, f"day {n} was not assigned a module"


def test_stakes_come_from_the_curriculums_own_type_field():
    assert curriculum.stakes(31) == 1.00  # CAPSTONE
    assert curriculum.stakes(7) == 0.95  # AI_CORE
    assert curriculum.stakes(1) == 0.15  # SETUP
    assert curriculum.stakes(1) < curriculum.stakes(10)


def test_setup_days_are_not_worth_interviewing_on():
    assert curriculum.day(1).is_interviewable is False
    assert curriculum.day(10).is_interviewable is True


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def test_all_twenty_profiles_load():
    assert len(candidates) == 20
    assert {c.member.id for c in candidates} >= {"CAND-001", "CAND-010", "CAND-018"}


def test_absent_missions_are_unobserved_not_skipped():
    """The mission list is a sample. Sarah lists 10 missions but completed 30 —
    treating the other 21 days as gaps would invent her a fake weak profile."""
    sarah = build_profile(by_id("CAND-001"))

    assert sarah.candidate.signals.missionsCompleted == 30
    assert len(sarah.records) == 10

    assert sarah.state_of(29) is MissionState.SKIPPED  # explicitly skipped
    assert sarah.state_of(2) is MissionState.UNOBSERVED  # simply not listed
    assert 2 not in sarah.skipped_days
    assert sarah.skipped_days == [29]


def test_failures_are_distinct_from_skips():
    """Gerald actually failed three missions. That is the loudest signal in the
    dataset and must never be blurred into 'skipped'."""
    gerald = build_profile(by_id("CAND-010"))
    assert gerald.failed_days == [8, 10, 22]
    assert gerald.skipped_days == [27, 28]
    assert 8 not in gerald.skipped_days


def test_first_try_ratio_separates_fluency_from_grinding():
    """Diane and Tyler both completed 31 missions. One never missed; the other
    brute-forced every one. Identical headline, opposite interviews."""
    diane = build_profile(by_id("CAND-018"))
    tyler = build_profile(by_id("CAND-017"))

    assert diane.candidate.signals.missionsCompleted == 31
    assert tyler.candidate.signals.missionsCompleted == 31

    assert diane.first_try_ratio == pytest.approx(1.0)
    assert tyler.first_try_ratio == pytest.approx(1 / 31, abs=0.01)
    assert diane.first_try_ratio > tyler.first_try_ratio


def test_struggle_is_detected_from_attempts():
    tyler = build_profile(by_id("CAND-017"))
    # Tyler passed almost everything, but on the fifth attempt.
    assert 7 in tyler.struggled_days
    assert tyler.clean_days == []

    diane = build_profile(by_id("CAND-018"))
    assert diane.struggled_days == []
    assert len(diane.clean_days) == 10


def test_seniority_and_avoidance_are_both_visible():
    """Harold is a Distinguished Engineer with 28 years who skipped both
    fine-tuning days and needed five attempts at MCP. Both facts must survive
    normalization — together they are the interview."""
    harold = build_profile(by_id("CAND-008"))
    assert harold.years == 28
    assert "Distinguished" in harold.role
    assert harold.skipped_days == [14, 15]
    assert harold.records[23].attempts == 5


def test_titles_differ_between_files_so_we_join_on_day_number():
    """candidates.json calls day 21 'LangChain Agents'; curriculum.json calls it
    'Agentic Frameworks: LangChain Agents & Tool Use'."""
    zara = build_profile(by_id("CAND-009"))
    assert zara.records[21].title == "LangChain Agents"
    assert curriculum.day(21).title != zara.records[21].title
    # Joining on day number still resolves correctly.
    assert curriculum.day(21).module_title == "Agentic AI & MCP"


def test_malformed_input_never_raises():
    from app.normalize import normalize_curriculum

    for junk in [None, {}, {"days": "nope"}, {"modules": [1, 2]}, []]:
        c = normalize_curriculum(junk)
        assert c.days == {}
