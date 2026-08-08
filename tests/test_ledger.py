from __future__ import annotations

import pytest

from app.ledger import (
    Claim,
    ClaimSource,
    ClaimStatus,
    Evidence,
    apply_evidence,
    binary_entropy,
    expected_information_gain,
    select_target,
    summarise,
)


def make(day: int, belief: float = 0.5, stakes: float = 0.8, source=ClaimSource.PROFILE_PRIOR):
    return Claim(
        day=day,
        concept=f"topic-{day}",
        source=source,
        prior=belief,
        belief=belief,
        stakes=stakes,
    )


def ev(turn: int, signal: float, weight: float = 1.0):
    return Evidence(turn=turn, signal=signal, weight=weight, note="")


# ---------------------------------------------------------------------------


def test_belief_moves_in_the_direction_of_evidence():
    c = make(8, belief=0.5)
    up = apply_evidence(c, ev(1, +0.9))
    down = apply_evidence(c, ev(1, -0.9))
    assert up.belief > 0.5
    assert down.belief < 0.5
    assert c.belief == 0.5, "input claim must not be mutated"


def test_beliefs_never_become_unfalsifiable():
    c = make(8, belief=0.5)
    for t in range(30):
        c = apply_evidence(c, ev(t, +1.0))
    assert c.belief <= 0.98

    c = make(8, belief=0.5)
    for t in range(30):
        c = apply_evidence(c, ev(t, -1.0))
    assert c.belief >= 0.02


def test_log_odds_makes_confident_beliefs_stubborn():
    """The same evidence should move an uncertain belief far more than a
    settled one — that asymmetry is the point of working in log-odds."""
    uncertain = apply_evidence(make(1, belief=0.50), ev(1, +0.8))
    confident = apply_evidence(make(2, belief=0.95), ev(1, +0.8))
    assert (uncertain.belief - 0.50) > (confident.belief - 0.95)


def test_entropy_peaks_at_maximum_uncertainty():
    assert binary_entropy(0.5) == pytest.approx(1.0)
    assert binary_entropy(0.95) < 0.3
    assert binary_entropy(0.05) < 0.3


def test_gain_prefers_uncertainty_then_decays_with_probing():
    uncertain = make(1, belief=0.5)
    settled = make(2, belief=0.95)
    assert expected_information_gain(uncertain) > expected_information_gain(settled)

    probed = apply_evidence(uncertain, ev(1, 0.0))
    assert expected_information_gain(probed) < expected_information_gain(uncertain)


def test_role_implied_claims_outrank_plain_priors():
    plain = make(1, belief=0.5, source=ClaimSource.PROFILE_PRIOR)
    implied = make(2, belief=0.5, source=ClaimSource.ROLE_IMPLIED)
    asserted = make(3, belief=0.5, source=ClaimSource.INTERVIEW_ASSERTION)
    assert (
        expected_information_gain(asserted)
        > expected_information_gain(implied)
        > expected_information_gain(plain)
    )


# ---------------------------------------------------------------------------


def test_high_stakes_claim_is_not_settled_on_one_answer():
    c = apply_evidence(make(10, belief=0.5, stakes=0.9), ev(1, +0.95))
    assert c.belief > 0.7
    assert c.status is ClaimStatus.PROBING, "one answer is not a verdict"

    c = apply_evidence(c, ev(2, +0.8))
    assert c.status is ClaimStatus.SUPPORTED
    assert c.corroborated


def test_low_stakes_claim_may_settle_immediately():
    c = apply_evidence(make(1, belief=0.5, stakes=0.2), ev(1, +0.95))
    assert c.status is ClaimStatus.SUPPORTED


def test_two_probes_on_the_same_turn_do_not_count_as_corroboration():
    c = make(10, stakes=0.9)
    c = apply_evidence(c, ev(4, +0.9))
    c = apply_evidence(c, ev(4, +0.9))
    assert not c.corroborated


# ---------------------------------------------------------------------------


def test_coverage_lock_beats_information_gain():
    """A juicy uncertain claim on an already-covered day must lose to a duller
    claim on an unvisited one when questions are running out."""
    juicy = make(5, belief=0.5, stakes=1.0)
    dull = make(9, belief=0.9, stakes=0.6)

    t = select_target(
        [juicy, dull],
        days_covered=[5, 1, 2],
        questions_asked=7,
        min_questions=8,
        min_distinct_days=4,
    )
    assert t.claim is dull
    assert "Coverage lock" in t.reason


def test_gain_wins_when_there_is_room_to_spare():
    juicy = make(5, belief=0.5, stakes=1.0)
    dull = make(9, belief=0.9, stakes=0.6)
    t = select_target(
        [juicy, dull],
        days_covered=[5, 1, 2, 3],
        questions_asked=1,
        min_questions=8,
        min_distinct_days=4,
    )
    assert t.claim is juicy


def test_selection_returns_a_readable_reason():
    t = select_target(
        [make(7)], days_covered=[], questions_asked=0, min_questions=8, min_distinct_days=4
    )
    assert t.claim is not None
    assert len(t.reason) > 30
    assert "day 7" in t.reason


def test_empty_and_fully_excluded_pools_are_safe():
    assert select_target([], days_covered=[], questions_asked=0, min_questions=8, min_distinct_days=4).claim is None
    t = select_target(
        [make(7)],
        days_covered=[],
        questions_asked=0,
        min_questions=8,
        min_distinct_days=4,
        exclude_days={7},
    )
    assert t.claim is None


def test_interview_spreads_across_at_least_four_days():
    claims = [make(d, belief=0.5, stakes=0.8) for d in (7, 8, 10, 12, 16, 22, 23, 28)]
    covered: list[int] = []
    by_day = {c.day: c for c in claims}

    for turn in range(1, 9):
        t = select_target(
            list(by_day.values()),
            days_covered=covered,
            questions_asked=turn - 1,
            min_questions=8,
            min_distinct_days=5,
        )
        assert t.claim is not None
        by_day[t.claim.day] = apply_evidence(t.claim, ev(turn, +0.4))
        covered.append(t.claim.day)

    assert len(set(covered)) >= 5


def test_summarise_reports_the_shape_of_the_ledger():
    claims = [make(1), apply_evidence(make(2, stakes=0.2), ev(1, +0.95))]
    s = summarise(claims)
    assert s["total"] == 2
    assert s["tested"] == 1
    assert s["supported"] == 1
    assert s["unprobed"] == 1
