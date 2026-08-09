"""Session state and store.

The spec hands us the `sessionId`; we never mint one. State is logically
in-process — no database was asked for and a restart losing a live interview
is acceptable — but persisting it is still worth doing when a real database
is one env var away: it means a backend redeploy or restart mid-interview
does not silently drop the candidate. When `SUPABASE_URL` /
`SUPABASE_SERVICE_ROLE_KEY` are present the store below persists to a
Supabase Postgres table via its PostgREST REST API; otherwise it falls back
to the original in-process dict unchanged, which is exactly what local
development and the test suite use.

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
# is used for the Supabase path because a Session graph is
# dataclasses-of-dataclasses with enums and a pydantic Candidate inside it —
# round-tripping that through JSON correctly means reimplementing pickle
# badly. We control both the write and the read side of this data (it is
# never attacker-supplied), so pickle's usual deserialization risk does not
# apply here. The service-role key bypasses Row Level Security by design —
# it must only ever live in the backend host's environment, never reach the
# frontend.
#
# Table (run once in the Supabase SQL editor):
#
#   create table if not exists sessions (
#     id text primary key,
#     payload text not null,
#     updated_at timestamptz not null default now()
#   );
#   create index if not exists sessions_updated_at_idx on sessions (updated_at);

_SESSIONS: dict[str, Session] = {}

_SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
_SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
_SUPABASE_TABLE = "sessions"


def _supabase_enabled() -> bool:
    return bool(_SUPABASE_URL and _SUPABASE_KEY)


def _supabase_client():
    import httpx

    return httpx.Client(
        base_url=f"{_SUPABASE_URL}/rest/v1",
        headers={
            "apikey": _SUPABASE_KEY,
            "Authorization": f"Bearer {_SUPABASE_KEY}",
            "Content-Type": "application/json",
        },
        timeout=10.0,
    )


def _encode(session: Session) -> str:
    return base64.b64encode(pickle.dumps(session)).decode("ascii")


def _decode(raw: str) -> Session:
    return pickle.loads(base64.b64decode(raw))


def get(session_id: str) -> Session | None:
    if not _supabase_enabled():
        return _SESSIONS.get(session_id)
    try:
        with _supabase_client() as c:
            r = c.get(f"/{_SUPABASE_TABLE}", params={"id": f"eq.{session_id}", "select": "payload"})
            r.raise_for_status()
            rows = r.json()
            return _decode(rows[0]["payload"]) if rows else None
    except Exception:
        log.exception("Supabase get failed for session %s", session_id)
        return None


def put(session: Session) -> Session:
    if not _supabase_enabled():
        _SESSIONS[session.id] = session
        return session
    try:
        with _supabase_client() as c:
            r = c.post(
                f"/{_SUPABASE_TABLE}",
                headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
                json=[{"id": session.id, "payload": _encode(session), "updated_at": _now()}],
            )
            r.raise_for_status()
    except Exception:
        log.exception("Supabase put failed for session %s", session.id)
    return session


def drop(session_id: str) -> bool:
    if not _supabase_enabled():
        return _SESSIONS.pop(session_id, None) is not None
    try:
        with _supabase_client() as c:
            r = c.delete(
                f"/{_SUPABASE_TABLE}",
                params={"id": f"eq.{session_id}"},
                headers={"Prefer": "return=representation"},
            )
            r.raise_for_status()
            return len(r.json()) > 0
    except Exception:
        log.exception("Supabase drop failed for session %s", session_id)
        return False


def count() -> int:
    """Exact in-process; exact under Supabase too, via PostgREST's Content-Range
    count header — no need to fetch rows just to count them."""
    if not _supabase_enabled():
        return len(_SESSIONS)
    try:
        with _supabase_client() as c:
            r = c.get(
                f"/{_SUPABASE_TABLE}",
                params={"select": "id"},
                headers={"Prefer": "count=exact", "Range": "0-0"},
            )
            r.raise_for_status()
            content_range = r.headers.get("content-range", "")  # e.g. "0-0/42"
            return int(content_range.split("/")[-1]) if "/" in content_range else -1
    except Exception:
        log.exception("Supabase count failed")
        return -1


def clear() -> None:
    """In-process only — tests and local scripts. Deliberately does not
    touch the Supabase table in production: this is a shared, durable store,
    not a per-test scratch space, and there is no safe indiscriminate wipe
    here."""
    _SESSIONS.clear()
