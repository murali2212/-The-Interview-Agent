"""Curriculum and candidate loading.

Two things in the supplied data will silently corrupt an interview if taken at
face value, and both are handled here:

1. `modules[].days` is a RANGE, not a list. `[7, 10]` means days 7,8,9,10.
   Reading it as a membership list loses days 8 and 9 from module 3.

2. A candidate's `missions` array is a SAMPLE. Profiles list ten missions while
   `signals.missionsCompleted` says thirty. A day that is absent is UNOBSERVED,
   not skipped — inventing a gap for every unlisted day would fabricate about
   twenty false weaknesses per candidate.

The two files also disagree on titles (curriculum day 21 is "Agentic
Frameworks: LangChain Agents & Tool Use"; the candidate file calls it "LangChain
Agents"). Everything here joins on the day NUMBER and never on the title.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .config import DATA_DIR
from .models import Candidate, Mission

# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Module:
    n: int
    title: str
    first_day: int
    last_day: int

    @property
    def days(self) -> list[int]:
        return list(range(self.first_day, self.last_day + 1))

    def contains(self, day: int) -> bool:
        return self.first_day <= day <= self.last_day


@dataclass(frozen=True)
class Day:
    day: int
    title: str
    type: str
    tools: tuple[str, ...]
    objectives: tuple[str, ...]
    module_n: int
    module_title: str

    @property
    def is_interviewable(self) -> bool:
        """Setup days test whether someone can install Python. Nobody's
        engineering judgement is revealed by day 1, so we do not spend
        questions there unless there is nothing else to ask about."""
        return self.type not in {"SETUP"}


#: How much a day's material is worth interviewing on, derived from the
#: curriculum's own `type` field rather than hand-tuned per day.
TYPE_WEIGHT: dict[str, float] = {
    "SETUP": 0.15,
    "LEARN": 0.60,
    "BUILD": 0.75,
    "AI_CORE": 0.95,
    "OPTIMIZE": 0.80,
    "SHIP_IT": 0.90,
    "CAPSTONE": 1.00,
}


@dataclass
class Curriculum:
    cohort: str
    modules: list[Module]
    days: dict[int, Day]

    def day(self, n: int) -> Day | None:
        return self.days.get(n)

    def module_for(self, day: int) -> Module | None:
        for m in self.modules:
            if m.contains(day):
                return m
        return None

    def stakes(self, day: int) -> float:
        d = self.days.get(day)
        if d is None:
            return 0.5
        return TYPE_WEIGHT.get(d.type.upper(), 0.7)

    @property
    def day_numbers(self) -> list[int]:
        return sorted(self.days)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _clean_strings(values: Iterable[Any]) -> tuple[str, ...]:
    out: list[str] = []
    for v in values:
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return tuple(out)


def normalize_curriculum(raw: Any) -> Curriculum:
    """Total: never raises on malformed input, drops what it cannot use."""
    if not isinstance(raw, dict):
        return Curriculum(cohort="", modules=[], days={})

    modules: list[Module] = []
    for m in _as_list(raw.get("modules")):
        if not isinstance(m, dict):
            continue
        span = [d for d in _as_list(m.get("days")) if isinstance(d, int)]
        if not span:
            continue
        # A range of one day may legitimately arrive as [5] or [5, 5].
        first, last = min(span), max(span)
        modules.append(
            Module(
                n=int(m.get("n", len(modules) + 1)),
                title=str(m.get("title", "")).strip(),
                first_day=first,
                last_day=last,
            )
        )

    days: dict[int, Day] = {}
    for d in _as_list(raw.get("days")):
        if not isinstance(d, dict):
            continue
        n = d.get("day")
        if not isinstance(n, int):
            continue
        module = next((m for m in modules if m.contains(n)), None)
        days[n] = Day(
            day=n,
            title=str(d.get("title", f"Day {n}")).strip(),
            type=str(d.get("type", "BUILD")).strip().upper(),
            tools=_clean_strings(_as_list(d.get("tools"))),
            objectives=_clean_strings(_as_list(d.get("objectives"))),
            module_n=module.n if module else 0,
            module_title=module.title if module else "",
        )

    return Curriculum(cohort=str(raw.get("cohort", "")).strip(), modules=modules, days=days)


# ---------------------------------------------------------------------------
# Candidate mission states
# ---------------------------------------------------------------------------


class MissionState(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    UNOBSERVED = "unobserved"


@dataclass(frozen=True)
class MissionRecord:
    day: int
    title: str
    state: MissionState
    attempts: int | None

    @property
    def struggled(self) -> bool:
        """Passed, but only after grinding. Genuine but shallow understanding
        is the most common thing this reveals."""
        return self.state is MissionState.PASSED and (self.attempts or 1) >= 4


def classify(mission: Mission) -> MissionState:
    if mission.skipped:
        return MissionState.SKIPPED
    if mission.passed is True:
        return MissionState.PASSED
    if mission.passed is False:
        return MissionState.FAILED
    return MissionState.UNOBSERVED


@dataclass
class Profile:
    """Everything the panel knows before a word is spoken."""

    candidate: Candidate
    records: dict[int, MissionRecord] = field(default_factory=dict)

    # --- identity ---------------------------------------------------------
    @property
    def name(self) -> str:
        return self.candidate.member.name or "the candidate"

    @property
    def role(self) -> str:
        return self.candidate.member.jobRole or ""

    @property
    def years(self) -> int:
        return int(self.candidate.member.yearsExperience or 0)

    # --- telemetry --------------------------------------------------------
    @property
    def first_try_ratio(self) -> float:
        """The most informative number in the whole dataset. Two candidates
        can both finish 31 missions while one never missed and the other
        needed five attempts every time."""
        s = self.candidate.signals
        if not s.missionsCompleted:
            return 0.0
        return max(0.0, min(1.0, s.missionsFirstTry / s.missionsCompleted))

    @property
    def consistency(self) -> float:
        """Commit days out of 31 — did they work steadily or cram."""
        return max(0.0, min(1.0, (self.candidate.signals.commitDays or 0) / 31))

    @property
    def completion(self) -> float:
        return max(0.0, min(1.0, (self.candidate.signals.missionsCompleted or 0) / 31))

    # --- mission views ----------------------------------------------------
    def state_of(self, day: int) -> MissionState:
        rec = self.records.get(day)
        return rec.state if rec else MissionState.UNOBSERVED

    def days_in(self, *states: MissionState) -> list[int]:
        wanted = set(states)
        return sorted(d for d, r in self.records.items() if r.state in wanted)

    @property
    def failed_days(self) -> list[int]:
        """Rare and loud. An explicit failure is the strongest negative signal
        the dataset offers and the highest-value thing to interview on."""
        return self.days_in(MissionState.FAILED)

    @property
    def skipped_days(self) -> list[int]:
        return self.days_in(MissionState.SKIPPED)

    @property
    def passed_days(self) -> list[int]:
        return self.days_in(MissionState.PASSED)

    @property
    def struggled_days(self) -> list[int]:
        return sorted(d for d, r in self.records.items() if r.struggled)

    @property
    def clean_days(self) -> list[int]:
        return sorted(
            d
            for d, r in self.records.items()
            if r.state is MissionState.PASSED and (r.attempts or 1) == 1
        )


def build_profile(candidate: Candidate) -> Profile:
    records: dict[int, MissionRecord] = {}
    for m in candidate.missions:
        if not isinstance(m.day, int):
            continue
        records[m.day] = MissionRecord(
            day=m.day,
            title=(m.title or "").strip(),
            state=classify(m),
            attempts=m.attempts,
        )
    return Profile(candidate=candidate, records=records)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def load_curriculum(path: str | Path | None = None) -> Curriculum:
    return normalize_curriculum(_read_json(Path(path) if path else DATA_DIR / "curriculum.json"))


def load_candidates(path: str | Path | None = None) -> list[Candidate]:
    """Only used for local demos and tests. In production the evaluator posts
    the candidate object to us; we never look one up."""
    raw = _read_json(Path(path) if path else DATA_DIR / "candidates.json")
    items = raw.get("candidates", raw) if isinstance(raw, dict) else raw
    out: list[Candidate] = []
    for item in _as_list(items):
        try:
            out.append(Candidate.model_validate(item))
        except Exception:
            continue
    return out
