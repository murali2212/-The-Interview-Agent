"""Session state and store.

The spec hands us the `sessionId`; we never mint one. State is in-process,
which is the right scope for this brief — no database was asked for and a
restart losing a live interview is acceptable.

`history` (raw turns) and `claims` (structured belief) are deliberately
separate. Message history alone is what makes an "AI interviewer" drift into a
scripted questionnaire: the model re-reads the transcript and free-associates.
The ledger is what gives the interview a spine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .ledger import Claim
from .models import Candidate, Feedback
from .normalize import Profile


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Turn:
    n: int
    kind: str
    day: int
    concept: str
    question: str
    rationale: str
    derived_from: int | None = None
    answer: str | None = None
    scores: dict[str, Any] | None = None


@dataclass
class Event:
    seq: int
    at: str
    agent: str
    action: str
    detail: str


@dataclass
class Session:
    id: str
    candidate: Candidate
    profile: Profile
    claims: list[Claim] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    summary: str = ""
    patterns: list[str] = field(default_factory=list)
    headline: str = ""
    done: bool = False
    feedback: Feedback | None = None
    created_at: str = field(default_factory=_now)

    # -- helpers -----------------------------------------------------------

    @property
    def pending(self) -> Turn | None:
        """The question that has been asked but not yet answered."""
        if self.turns and self.turns[-1].answer is None:
            return self.turns[-1]
        return None

    @property
    def answered(self) -> list[Turn]:
        return [t for t in self.turns if t.answer is not None and t.scores is not None]

    @property
    def days_covered(self) -> list[int]:
        return sorted({t.day for t in self.turns})

    def claim_for(self, day: int) -> Claim | None:
        return next((c for c in self.claims if c.day == day), None)

    def replace_claim(self, updated: Claim) -> None:
        self.claims = [updated if c.day == updated.day else c for c in self.claims]

    def log(self, agent: str, action: str, detail: str) -> None:
        self.events.append(
            Event(seq=len(self.events) + 1, at=_now(), agent=agent, action=action, detail=detail)
        )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_SESSIONS: dict[str, Session] = {}


def get(session_id: str) -> Session | None:
    return _SESSIONS.get(session_id)


def put(session: Session) -> Session:
    _SESSIONS[session.id] = session
    return session


def drop(session_id: str) -> bool:
    return _SESSIONS.pop(session_id, None) is not None


def count() -> int:
    return len(_SESSIONS)


def clear() -> None:
    _SESSIONS.clear()
