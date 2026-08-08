"""Wire contract.

This module is a direct transcription of `docs-technical-spec.md`. It is the
ONLY place the organizers' request/response shapes appear. Everything deeper in
the app speaks our own domain types, so if the spec moves, it moves here.

Spec summary:
    POST /api/interview          (single endpoint, no auth)

    start turn ->  {"sessionId": "...", "candidate": {...}}
    later turns -> {"sessionId": "...", "message": "..."}
    response    -> {"reply": "...", "done": false}
    final       -> {"reply": "...", "done": true, "feedback": {...}}
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Candidate — posted to us in the first request. We do NOT look this up.
# ---------------------------------------------------------------------------


class Member(BaseModel):
    """Who the candidate is. `jobRole` and `yearsExperience` are the closest
    thing this dataset has to a self-declared strength, so they matter."""

    model_config = ConfigDict(extra="allow")

    id: str = ""
    name: str = ""
    jobRole: str = ""
    yearsExperience: int = 0
    education: str = ""
    status: str = ""


class Mission(BaseModel):
    """One mission record.

    Four states are possible and they are NOT interchangeable:

        passed=True                -> observed success (weight by attempts)
        passed=False               -> observed failure (the loudest signal here)
        skipped=True               -> deliberate avoidance
        absent from the list       -> UNOBSERVED, say nothing about it

    The mission list is a SAMPLE: profiles routinely list ten missions while
    `signals.missionsCompleted` says thirty. Treating an unlisted day as a gap
    would invent roughly twenty false weaknesses per candidate.
    """

    model_config = ConfigDict(extra="allow")

    day: int
    title: str = ""
    passed: bool | None = None
    attempts: int | None = None
    skipped: bool | None = None


class Signals(BaseModel):
    """Cohort telemetry.

    `missionsFirstTry / missionsCompleted` is the single most useful ratio in
    the dataset: two candidates can both complete 31 missions while one never
    missed and the other needed five attempts every time.
    """

    model_config = ConfigDict(extra="allow")

    commitDays: int = 0
    missionsCompleted: int = 0
    missionsFirstTry: int = 0


class Candidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    member: Member = Field(default_factory=Member)
    missions: list[Mission] = Field(default_factory=list)
    signals: Signals = Field(default_factory=Signals)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class InterviewRequest(BaseModel):
    """Both request shapes in one model — the spec uses one endpoint for both.

    Routing rule: `candidate` present -> start a session.
                  `message` present   -> continue that session.
    """

    model_config = ConfigDict(extra="allow")

    sessionId: str
    candidate: Candidate | None = None
    message: str | None = None

    def is_start(self) -> bool:
        return self.candidate is not None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class Feedback(BaseModel):
    """Required final-turn payload. Field names are fixed by the spec —
    `next` shadows a Python builtin but the wire name is not ours to change."""

    summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    next: list[str] = Field(default_factory=list)


class InterviewResponse(BaseModel):
    """`reply` and `done` are the contract. `insight` is ours.

    The spec fixes what must be present, not what may be. Everything the
    console renders — the evidence ledger, the calibration table, the agent
    trace — rides in `insight`, which an evaluator can ignore entirely.

    `reply` must stay plain speakable prose: no markdown, no bullet lists, no
    code fences. The voice transport reads this string aloud verbatim, and
    asterisks get pronounced.
    """

    model_config = ConfigDict(extra="allow")

    reply: str
    done: bool = False
    feedback: Feedback | None = None
    insight: dict[str, Any] | None = None
