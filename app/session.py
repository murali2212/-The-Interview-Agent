"""Session state and store.

The spec hands us the `sessionId`; we never mint one. State is logically
in-process — no database was asked for and a restart losing a live interview
is acceptable — but *where* "in-process" means something depends on how this
runs. A local `uvicorn` process lives for the whole interview, so a plain
dict is correct there. A Vercel serverless deployment has no such guarantee:
each request can land on a different, short-lived instance, so a plain dict
would silently lose the session between question 1 and question 2. When
`KV_REST_API_URL` / `KV_REST_API_TOKEN` are present (Vercel KV, which is
Upstash Redis under a REST API) the store below persists there instead;
otherwise it falls back to the original in-process dict unchanged.

`history` (raw turns) and `claims` (structured belief) are deliberately
separate. Message history alone is what makes an "AI interviewer" drift into a
scripted questionnaire: the model re-reads the transcript and free-associates.
The ledger is what gives the interview a spine.
"""

from __future__ import annotations

import base64
import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .ledger import Claim
from .models import Candidate, Feedback
from .normalize import Profile

log = logging.getLogger("abtalks.session")


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
#
# Two backends behind the same four functions. Pickle (not hand-rolled JSON)
# is used for the KV path because a Session graph is dataclasses-of-dataclasses
# with enums and a pydantic Candidate inside it — round-tripping that through
# JSON correctly means reimplementing pickle badly. We control both the write
# and the read side of this data (it is never attacker-supplied), so pickle's
# usual deserialization risk does not apply here.

_SESSIONS: dict[str, Session] = {}

_KV_URL = os.getenv("KV_REST_API_URL", "").rstrip("/")
_KV_TOKEN = os.getenv("KV_REST_API_TOKEN", "").strip()
_KV_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "7200"))  # 2h — an abandoned interview should not live forever
_KV_PREFIX = "abtalks:session:"


def _kv_enabled() -> bool:
    return bool(_KV_URL and _KV_TOKEN)


def _kv_client():
    import httpx

    return httpx.Client(base_url=_KV_URL, headers={"Authorization": f"Bearer {_KV_TOKEN}"}, timeout=10.0)


def _encode(session: Session) -> str:
    return base64.b64encode(pickle.dumps(session)).decode("ascii")


def _decode(raw: str) -> Session:
    return pickle.loads(base64.b64decode(raw))


def get(session_id: str) -> Session | None:
    if not _kv_enabled():
        return _SESSIONS.get(session_id)
    try:
        with _kv_client() as c:
            r = c.get(f"/get/{_KV_PREFIX}{session_id}")
            r.raise_for_status()
            raw = r.json().get("result")
            return _decode(raw) if raw else None
    except Exception:
        log.exception("KV get failed for session %s", session_id)
        return None


def put(session: Session) -> Session:
    if not _kv_enabled():
        _SESSIONS[session.id] = session
        return session
    try:
        with _kv_client() as c:
            r = c.post(
                f"/set/{_KV_PREFIX}{session.id}",
                params={"EX": _KV_TTL_SECONDS},
                content=_encode(session),
            )
            r.raise_for_status()
    except Exception:
        log.exception("KV put failed for session %s", session.id)
    return session


def drop(session_id: str) -> bool:
    if not _kv_enabled():
        return _SESSIONS.pop(session_id, None) is not None
    try:
        with _kv_client() as c:
            r = c.post(f"/del/{_KV_PREFIX}{session_id}")
            r.raise_for_status()
            return bool(r.json().get("result"))
    except Exception:
        log.exception("KV drop failed for session %s", session_id)
        return False


def count() -> int:
    """Best-effort. Exact in-process; approximate (whole-DB DBSIZE) under KV,
    which is fine since this only feeds the informational /api/health field."""
    if not _kv_enabled():
        return len(_SESSIONS)
    try:
        with _kv_client() as c:
            r = c.post("/dbsize")
            r.raise_for_status()
            return int(r.json().get("result", 0))
    except Exception:
        log.exception("KV count failed")
        return -1


def clear() -> None:
    """In-process only — tests and local scripts. Deliberately does not
    flush the shared KV store in production: this is a per-key store, not a
    dedicated namespace, and there is no safe indiscriminate wipe here."""
    _SESSIONS.clear()
