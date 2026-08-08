"""The Evidence Ledger — the interview policy.

A scripted interviewer walks a list. This one holds explicit, falsifiable
beliefs about the candidate and spends each question on whichever belief it is
least certain about and most cares to resolve.

    gain(claim) = H(belief) x stakes x decay(times_probed) x source_bonus

`H` is binary entropy, so uncertainty peaks at belief 0.5: the panel is pulled
towards what it genuinely does not know rather than what is easy to ask.

Beliefs move in LOG-ODDS space, which is why a strong answer moves a 0.5 belief
a long way and a 0.95 belief barely at all — the same asymmetry a human
interviewer has.

Claims are keyed by curriculum DAY. That is the unit the cohort teaches in and
the unit the mission telemetry reports in, so anything finer would be inventing
structure the data does not have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

UPDATE_K = 1.6
BELIEF_FLOOR = 0.02
BELIEF_CEILING = 0.98
SUPPORTED_AT = 0.70
REFUTED_AT = 0.30
PROBE_DECAY = 0.8

#: A high-stakes claim may not be declared settled on one answer. The belief
#: still moves, but the status stays `PROBING` so nothing downstream reports it
#: as resolved and the director keeps the question open.
CORROBORATION_STAKES = 0.60


class ClaimSource(str, Enum):
    PROFILE_PRIOR = "profile-prior"
    #: This dataset has no self-reported strengths, so the closest equivalent is
    #: what the job title implies. A Distinguished Engineer who skipped
    #: fine-tuning is making a claim whether or not they typed one.
    ROLE_IMPLIED = "role-implied"
    #: Something they asserted mid-interview. Testing it while it is live is
    #: the highest-value thing available.
    INTERVIEW_ASSERTION = "interview-assertion"


class ClaimStatus(str, Enum):
    UNPROBED = "unprobed"
    PROBING = "probing"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    PARTIAL = "partial"


@dataclass(frozen=True)
class Evidence:
    turn: int
    signal: float  # -1..+1
    weight: float  # 0..1
    note: str


@dataclass
class Claim:
    day: int
    concept: str
    source: ClaimSource
    prior: float
    belief: float
    stakes: float
    status: ClaimStatus = ClaimStatus.UNPROBED
    evidence: list[Evidence] = field(default_factory=list)
    why_prior: str = ""

    @property
    def id(self) -> str:
        return f"d{self.day}"

    @property
    def delta(self) -> float:
        return self.belief - self.prior

    @property
    def probes(self) -> int:
        return len(self.evidence)

    @property
    def corroborated(self) -> bool:
        return len({e.turn for e in self.evidence}) >= 2


# ---------------------------------------------------------------------------
# Maths
# ---------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_belief(value: float) -> float:
    return clamp(value, BELIEF_FLOOR, BELIEF_CEILING)


def logit(p: float) -> float:
    p = clamp_belief(p)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def binary_entropy(p: float) -> float:
    p = clamp(p, 1e-9, 1 - 1e-9)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def status_for(claim: Claim) -> ClaimStatus:
    if not claim.evidence:
        return ClaimStatus.UNPROBED

    if claim.belief >= SUPPORTED_AT:
        settled = ClaimStatus.SUPPORTED
    elif claim.belief <= REFUTED_AT:
        settled = ClaimStatus.REFUTED
    else:
        settled = ClaimStatus.PARTIAL

    # One answer is not a verdict.
    if (
        settled in (ClaimStatus.SUPPORTED, ClaimStatus.REFUTED)
        and claim.stakes >= CORROBORATION_STAKES
        and not claim.corroborated
    ):
        return ClaimStatus.PROBING
    if len(claim.evidence) == 1 and settled is ClaimStatus.PARTIAL:
        return ClaimStatus.PROBING
    return settled


def apply_evidence(claim: Claim, ev: Evidence) -> Claim:
    """Bayesian-style update. Returns a NEW claim; never mutates the input."""
    signal = clamp(ev.signal, -1.0, 1.0)
    weight = clamp(ev.weight, 0.0, 1.0)
    moved = clamp_belief(sigmoid(logit(claim.belief) + UPDATE_K * signal * weight))

    updated = replace(
        claim,
        belief=moved,
        evidence=[*claim.evidence, replace(ev, signal=signal, weight=weight)],
    )
    return replace(updated, status=status_for(updated))


SOURCE_BONUS: dict[ClaimSource, float] = {
    ClaimSource.PROFILE_PRIOR: 1.00,
    ClaimSource.ROLE_IMPLIED: 1.25,
    ClaimSource.INTERVIEW_ASSERTION: 1.35,
}


def expected_information_gain(claim: Claim) -> float:
    gain = binary_entropy(claim.belief) * claim.stakes
    gain *= 1.0 / (1.0 + PROBE_DECAY * claim.probes)
    gain *= SOURCE_BONUS.get(claim.source, 1.0)
    return gain


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclass
class Target:
    claim: Claim | None
    reason: str


def select_target(
    claims: list[Claim],
    *,
    days_covered: list[int],
    questions_asked: int,
    min_questions: int,
    min_distinct_days: int,
    exclude_days: set[int] | None = None,
) -> Target:
    """Pick the claim worth resolving next.

    Breadth is a hard constraint, not a preference: if the questions remaining
    are only just enough to reach `min_distinct_days`, an uncovered day wins
    even when its information gain is lower.
    """
    exclude = exclude_days or set()
    pool = [c for c in claims if c.day not in exclude]
    if not pool:
        return Target(None, "No claims remain to test.")

    covered = set(days_covered)
    uncovered = [c for c in pool if c.day not in covered]
    days_still_needed = max(0, min_distinct_days - len(covered))
    questions_left = max(0, min_questions - questions_asked)

    # --- coverage lock ----------------------------------------------------
    if uncovered and days_still_needed >= questions_left and days_still_needed > 0:
        pick = max(uncovered, key=expected_information_gain)
        return Target(
            pick,
            f"Coverage lock: {questions_left} question(s) left and {days_still_needed} "
            f"curriculum day(s) still unvisited, so day {pick.day} ({pick.concept}) is taken "
            f"now. Breadth outranks information gain here.",
        )

    # --- corroborate before concluding ------------------------------------
    thin = [
        c
        for c in pool
        if c.probes == 1 and c.stakes >= CORROBORATION_STAKES and not c.corroborated
    ]
    if thin and len(covered) >= min_distinct_days:
        pick = max(thin, key=lambda c: c.stakes)
        return Target(
            pick,
            f"{pick.concept} (day {pick.day}) is carrying a high-stakes verdict on a single "
            f"answer. Going back for a second, independent read before the report commits.",
        )

    # --- otherwise, maximise information gain -----------------------------
    pick = max(pool, key=expected_information_gain)
    gain = expected_information_gain(pick)
    opened = "" if pick.day in covered else f" It also opens day {pick.day}, so far untouched."
    return Target(
        pick,
        f"Belief on {pick.concept} (day {pick.day}) sits at {pick.belief:.2f} after "
        f"{pick.probes} probe(s) — the highest remaining uncertainty on a high-stakes topic "
        f"(gain {gain:.2f}).{opened}",
    )


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------


def distinct_days(claims: list[Claim]) -> list[int]:
    return sorted({c.day for c in claims if c.evidence})


def summarise(claims: list[Claim]) -> dict[str, float | int]:
    counts = {s.value: 0 for s in ClaimStatus}
    for c in claims:
        counts[c.status.value] += 1
    tested = [c for c in claims if c.evidence]
    return {
        "total": len(claims),
        **counts,
        "tested": len(tested),
        "mean_belief": round(sum(c.belief for c in claims) / len(claims), 3) if claims else 0.0,
        "mean_shift": round(sum(abs(c.delta) for c in tested) / len(tested), 3) if tested else 0.0,
    }
