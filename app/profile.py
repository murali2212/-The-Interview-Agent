"""Profile Analyst — cohort telemetry becomes explicit, falsifiable priors.

This is what makes the interview personalised rather than generic. Before a
word is spoken the panel commits to a number for every interviewable day: how
likely is it that this person actually understands this material.

Three signals drive it:

1. **Mission outcome** — passed / failed / skipped, weighted by attempts.
   A pass on the fifth attempt is not the same event as a pass on the first.

2. **Fluency** — `missionsFirstTry / missionsCompleted`. Diane and Tyler both
   finished 31 missions; Diane never missed, Tyler needed five attempts every
   time. Same headline, opposite candidates.

3. **Role-implied expertise** — this dataset has no self-reported strengths, so
   the job title stands in. A Distinguished Engineer of 28 years who skipped
   both fine-tuning days is making a claim whether or not they typed one, and
   an unverified claim is the most expensive thing to leave in a report.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ledger import Claim, ClaimSource
from .normalize import Curriculum, Day, MissionState, Profile

# ---------------------------------------------------------------------------
# Priors from mission outcome
# ---------------------------------------------------------------------------

#: belief that the candidate genuinely understands the day's material
BASE_PRIOR: dict[MissionState, float] = {
    MissionState.PASSED: 0.78,
    MissionState.FAILED: 0.15,
    MissionState.SKIPPED: 0.22,
    MissionState.UNOBSERVED: 0.50,
}

#: How much a pass is discounted by the number of attempts it took.
ATTEMPT_PENALTY: dict[int, float] = {1: 0.10, 2: 0.00, 3: -0.08, 4: -0.18, 5: -0.26}

#: Unobserved days sit at belief ~0.5, which is exactly where binary entropy
#: peaks — so raw information gain finds them irresistible and the interview
#: drifts onto material the candidate never touched. The brief is explicit that
#: we assess "the concepts they have COMPLETED", so unobserved days are damped
#: hard. They stay on the board (they are legitimate ground when the record is
#: thin) but they no longer outbid work we can actually see.
UNOBSERVED_STAKES_FACTOR = 0.40

#: Except where the role implies they should own it anyway. "You are a DevOps
#: engineer and there is no sign you ever deployed this" is a real question.
UNOBSERVED_ROLE_IMPLIED_FACTOR = 0.80


def topic_of(day: Day) -> str:
    """A short, speakable label. Curriculum titles carry lecture-style
    prefixes ("Advanced Prompting: Function Calling...") that read badly when
    spoken aloud, so the part after the colon wins."""
    title = day.title
    if ":" in title:
        tail = title.split(":", 1)[1].strip()
        if len(tail) >= 8:
            title = tail
    for prefix in ("The ", "Building & Populating the ", "Building the "):
        if title.startswith(prefix):
            title = title[len(prefix) :]
    return title.strip()


# ---------------------------------------------------------------------------
# Role-implied expertise
# ---------------------------------------------------------------------------

#: keyword in job title -> curriculum days that role is expected to own
ROLE_EXPECTATIONS: list[tuple[tuple[str, ...], tuple[int, ...], str]] = [
    (("devops", "sre", "platform", "infrastructure"), (28, 29, 30), "deployment and operations"),
    (("data engineer", "data"), (4, 5, 6, 7, 9), "data pipelines and processing"),
    (("ai engineer", "ml", "machine learning", "artificial intelligence"),
     (7, 10, 11, 13, 21, 22, 23), "the AI core of the stack"),
    (("backend", "software engineer", "developer"), (16, 18, 20, 24), "backend services and APIs"),
    (("mobile",), (16, 17, 18, 19), "client integration"),
    (("ux", "design", "research"), (17, 19), "interface and presentation"),
    (("security",), (27,), "security and guardrails"),
    (("architect", "principal", "distinguished", "staff", "lead"),
     (22, 24, 25, 27, 30, 31), "system design and production judgement"),
    (("analyst", "business", "marketing", "hr", "support"), (), ""),
    (("intern", "junior", "student"), (), ""),
]

#: A senior engineer is expected to have opinions about production regardless
#: of their specialism.
SENIORITY_DAYS: tuple[int, ...] = (27, 28, 30, 31)


def role_expectations(profile: Profile) -> tuple[set[int], str]:
    role = (profile.role or "").lower()
    days: set[int] = set()
    labels: list[str] = []

    for keywords, expected, label in ROLE_EXPECTATIONS:
        if any(k in role for k in keywords):
            days.update(expected)
            if label:
                labels.append(label)
            break  # first match wins; titles are ordered most-specific first

    if profile.years >= 10:
        days.update(SENIORITY_DAYS)
        labels.append("production readiness, given seniority")

    return days, "; ".join(labels)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@dataclass
class Analysis:
    claims: list[Claim]
    patterns: list[str]
    headline: str


def _prior_for(profile: Profile, day: Day, curriculum: Curriculum) -> tuple[float, str]:
    state = profile.state_of(day.day)
    rec = profile.records.get(day.day)
    prior = BASE_PRIOR[state]
    why: list[str] = []

    if state is MissionState.PASSED and rec:
        attempts = rec.attempts or 1
        prior += ATTEMPT_PENALTY.get(min(attempts, 5), -0.26)
        why.append(f"passed in {attempts} attempt{'s' if attempts != 1 else ''}")
    elif state is MissionState.FAILED and rec:
        why.append(f"failed after {rec.attempts or 0} attempt(s)")
    elif state is MissionState.SKIPPED:
        why.append("skipped outright")
    else:
        # Unobserved: we know nothing specific, so we lean on how the candidate
        # performs overall rather than inventing a gap.
        prior = 0.35 + 0.30 * profile.first_try_ratio
        why.append("not in the mission record — inferred from overall performance")

    # Fluency nudges everything: someone who never needs a second attempt
    # probably understood the days we cannot see either.
    prior += 0.08 * (profile.first_try_ratio - 0.5)
    # Steady work beats cramming.
    prior += 0.04 * (profile.consistency - 0.5)

    return max(0.05, min(0.95, prior)), ", ".join(why)


def analyse(profile: Profile, curriculum: Curriculum, max_claims: int = 26) -> Analysis:
    expected_days, role_label = role_expectations(profile)
    claims: list[Claim] = []

    for day in curriculum.days.values():
        if not day.is_interviewable:
            continue

        prior, why = _prior_for(profile, day, curriculum)
        stakes = curriculum.stakes(day.day)
        source = ClaimSource.PROFILE_PRIOR
        unobserved = profile.state_of(day.day) is MissionState.UNOBSERVED

        if unobserved:
            stakes *= (
                UNOBSERVED_ROLE_IMPLIED_FACTOR
                if day.day in expected_days
                else UNOBSERVED_STAKES_FACTOR
            )

        if day.day in expected_days:
            source = ClaimSource.ROLE_IMPLIED
            # Their role asserts competence here, so we start from a higher
            # belief AND care more about resolving it.
            prior = max(prior, 0.72)
            stakes = min(1.0, stakes + 0.15)
            why = f"{profile.role} is expected to own this ({why})"

        claims.append(
            Claim(
                day=day.day,
                concept=topic_of(day),
                source=source,
                prior=prior,
                belief=prior,
                stakes=stakes,
                why_prior=why,
            )
        )

    # Keep the ledger tractable: the most consequential and least certain first.
    claims.sort(key=lambda c: c.stakes * (1 - abs(c.prior - 0.5)), reverse=True)
    claims = claims[:max_claims]
    claims.sort(key=lambda c: c.day)

    return Analysis(
        claims=claims,
        patterns=detect_patterns(profile, curriculum, expected_days),
        headline=headline(profile, role_label),
    )


def detect_patterns(
    profile: Profile, curriculum: Curriculum, expected_days: set[int]
) -> list[str]:
    """Plain-language observations. These steer the interviewer and appear in
    the trace, so they must be specific enough to act on."""
    out: list[str] = []
    ratio = profile.first_try_ratio

    if ratio >= 0.85:
        out.append(
            f"Passes almost everything first time ({profile.candidate.signals.missionsFirstTry}"
            f"/{profile.candidate.signals.missionsCompleted}) — probe for depth, not coverage."
        )
    elif ratio <= 0.20 and profile.candidate.signals.missionsCompleted >= 15:
        out.append(
            f"Completed a great deal but almost nothing first time "
            f"({profile.candidate.signals.missionsFirstTry}"
            f"/{profile.candidate.signals.missionsCompleted}) — persistence is proven, "
            f"understanding is not. Expect vocabulary without mechanism."
        )

    if profile.failed_days:
        titles = ", ".join(
            f"day {d} ({topic_of(curriculum.day(d))})"
            for d in profile.failed_days
            if curriculum.day(d)
        )
        out.append(f"Outright failures on {titles} — the loudest signal in the record.")

    if profile.skipped_days:
        out.append(
            "Skipped " + ", ".join(f"day {d}" for d in profile.skipped_days) + " deliberately."
        )

    unverified = sorted(
        d
        for d in expected_days
        if profile.state_of(d) in (MissionState.SKIPPED, MissionState.FAILED, MissionState.UNOBSERVED)
    )
    if unverified and profile.role:
        out.append(
            f"{profile.role} with {profile.years} years, yet day(s) "
            + ", ".join(str(d) for d in unverified)
            + " are unproven — the role claims ground the record does not support."
        )

    if profile.consistency <= 0.4:
        out.append(
            f"Worked on only {profile.candidate.signals.commitDays} of 31 days — "
            f"knowledge may be patchy rather than shallow."
        )

    return out


def headline(profile: Profile, role_label: str) -> str:
    bits = [f"{profile.name}"]
    if profile.role:
        bits.append(f"{profile.role}, {profile.years}y")
    s = profile.candidate.signals
    bits.append(f"{s.missionsCompleted}/31 missions, {s.missionsFirstTry} first-try")
    if role_label:
        bits.append(f"expected to own {role_label}")
    return " · ".join(bits)
