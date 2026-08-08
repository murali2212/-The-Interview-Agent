"""Audits the six graded requirements against a REAL interview.

Runs end to end through the HTTP endpoint with FastAPI's TestClient, then
inspects both the wire responses and the internal session. Prints PASS/FAIL
with the evidence that produced each verdict.

Run offline by default so a Groq rate limit cannot turn a structural check into
a false failure:

    python scripts/verify_requirements.py            # deterministic engine
    python scripts/verify_requirements.py --live     # hit Groq
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LIVE = "--live" in sys.argv
if not LIVE:
    os.environ["OFFLINE_ONLY"] = "true"

from fastapi.testclient import TestClient  # noqa: E402

from app import session as store  # noqa: E402
from app.api import app  # noqa: E402
from app.normalize import load_candidates, load_curriculum  # noqa: E402

GREEN, RED, DIM, BOLD, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, evidence: str) -> None:
    results.append((ok, name, evidence))
    mark = f"{GREEN}PASS{OFF}" if ok else f"{RED}FAIL{OFF}"
    print(f"  [{mark}] {name}")
    for line in evidence.splitlines():
        print(f"         {DIM}{line}{OFF}")


ANSWERS = [
    "We built the retrieval layer on Chroma locally. Roughly forty thousand chunks with "
    "eighty tokens of overlap, because at zero overlap answers were getting cut mid-clause "
    "and context recall sat around 0.62.",
    "The router checks whether the question is aggregate or lookup. Anything with a sum or a "
    "date range goes to SQL, everything else goes to vector search, and we merge and dedupe "
    "when both fire.",
    "Honestly I did not implement that part myself, so I would be guessing at the details.",
    "We added citations by keeping the chunk id on every retrieved record and forcing the "
    "model to quote the span it used. Unsupported claims dropped a lot but answered-rate fell "
    "from about 98 percent to 87.",
    "I built the whole Kubernetes deployment end to end in production and handled all the "
    "scaling, it worked really well at scale.",
    "Token cost was the driver. We cut the prompt from about 4k to 1.8k tokens by dropping "
    "duplicate context and cached repeated queries in Redis, which roughly halved latency.",
    "We wrapped each capability as a LangChain tool and let a ReAct agent pick. The reasoning "
    "trace showed it choosing the SQL tool for claims totals, which is what we wanted.",
    "Prompt injection was the main worry, so we validated input and never let tool output be "
    "treated as instructions. We tested a few jailbreak strings against it.",
    "We logged latency and tool calls with structured logging and watched failures in Grafana.",
    "The hardest part was retrieval quality. I would spend more time on evaluation earlier "
    "instead of tuning prompts blindly.",
    "We used Pydantic models to validate structured outputs before rendering them.",
    "I would add a proper regression suite before shipping anything else.",
]

print(f"\n{BOLD}ABTalks Interview Agent — requirement audit{OFF}")
print(f"{DIM}mode: {'LIVE (Groq)' if LIVE else 'offline deterministic engine'}{OFF}\n")

store.clear()
client = TestClient(app)
curriculum = load_curriculum()
candidate = next(c for c in load_candidates() if c.member.id == "CAND-005")
SID = "audit-session-1"

# ---------------------------------------------------------------------------
# 6. HTTP endpoint (checked first — everything else rides on it)
# ---------------------------------------------------------------------------
print(f"{BOLD}Requirement 6 — required HTTP endpoint{OFF}")

start = client.post(
    "/api/interview",
    json={"sessionId": SID, "candidate": candidate.model_dump()},
)
ok = start.status_code == 200
body = start.json() if ok else {}
check(
    ok and "reply" in body and body.get("done") is False,
    "POST /api/interview starts a session",
    f"status {start.status_code} · keys {sorted(body)[:5]} · done={body.get('done')}",
)

replies = [body.get("reply", "")]
final = None
for i, ans in enumerate(ANSWERS):
    r = client.post("/api/interview", json={"sessionId": SID, "message": ans})
    if r.status_code != 200:
        check(False, "conversation turn", f"turn {i + 1} returned {r.status_code}")
        break
    payload = r.json()
    replies.append(payload.get("reply", ""))
    if payload.get("done"):
        final = payload
        break

answers_given = len(replies) - 1
check(
    final is not None and answers_given == 8,
    "interview ends on the 8th answer and returns feedback in the same response",
    f"done=true arrived after {answers_given} answers"
    if final
    else "never terminated",
)

bad = client.post("/api/interview", json={"sessionId": "nope", "message": "hi"})
check(
    bad.status_code in (200, 400, 404),
    "unknown sessionId handled without a crash",
    f"status {bad.status_code}",
)

# ---------------------------------------------------------------------------
s = store.get(SID)
assert s is not None, "session vanished"
questions = [t for t in s.turns]
days = sorted({t.day for t in questions})

# ---------------------------------------------------------------------------
print(f"\n{BOLD}Requirement 2 — at least 8 questions across 4+ curriculum days{OFF}")
check(
    len(questions) == 8,
    f"asked exactly 8 questions (got {len(questions)})",
    " · ".join(f"Q{t.n}:{t.kind}" for t in questions),
)
check(
    len(days) >= 4,
    f"covered {len(days)} distinct curriculum days (floor 4)",
    "days "
    + ", ".join(f"{d} ({curriculum.day(d).title[:34]})" for d in days if curriculum.day(d)),
)

# Day count alone is gameable: days 27,28,29,30,31 are five distinct days and
# one corner of the curriculum. Real coverage means spanning modules.
modules = sorted({m.n for d in days if (m := curriculum.module_for(d))})
check(
    len(modules) >= 3,
    f"spans {len(modules)} distinct curriculum modules (floor 3)",
    "; ".join(
        f"module {n} ({next(m.title for m in curriculum.modules if m.n == n)})" for n in modules
    ),
)
check(
    max(days) - min(days) >= 8,
    "coverage is not clustered in consecutive days",
    f"day range {min(days)}–{max(days)}",
)

# The brief says assess "the concepts they have COMPLETED". Unobserved days sit
# at maximum entropy, so without damping the interview drifts onto material the
# candidate never touched.
from app.normalize import MissionState, build_profile  # noqa: E402

prof = build_profile(candidate)
observed = [d for d in days if prof.state_of(d) is not MissionState.UNOBSERVED]
check(
    len(observed) >= max(3, (len(days) + 1) // 2),
    "coverage concentrates on work the candidate actually did",
    f"{len(observed)}/{len(days)} covered days are in their mission record: "
    + ", ".join(f"{d}={prof.state_of(d).value}" for d in days),
)

# ---------------------------------------------------------------------------
print(f"\n{BOLD}Requirement 3 — follow-ups generated from previous responses{OFF}")
follow = [t for t in questions if t.derived_from is not None]
valid = all(
    t.derived_from < t.n and any(x.n == t.derived_from and x.answer for x in questions)
    for t in follow
)
check(
    len(follow) >= 1 and valid,
    f"{len(follow)} follow-up(s), each anchored to an earlier ANSWERED turn",
    "\n".join(f"Q{t.n} follows Q{t.derived_from}: {t.question[:90]}" for t in follow[:3])
    or "none",
)

adaptive = {t.kind for t in questions}
check(
    len(adaptive) >= 3,
    f"question strategy adapts ({len(adaptive)} distinct kinds)",
    ", ".join(sorted(adaptive)),
)

# ---------------------------------------------------------------------------
print(f"\n{BOLD}Requirement 4 — context maintained throughout{OFF}")
check(
    all(t.answer is not None for t in questions[:-1]),
    "every earlier turn retains its question, answer and scoring",
    f"{len(s.answered)} scored turns held in session state",
)
check(
    bool(s.summary),
    "rolling summary maintained across the interview",
    (s.summary[:150] + "…") if s.summary else "EMPTY",
)
evidence_ct = sum(len(c.evidence) for c in s.claims)
moved = [c for c in s.claims if c.evidence]
check(
    evidence_ct >= len(questions) - 1,
    "each answer updates persistent belief state",
    f"{evidence_ct} pieces of evidence across {len(moved)} claims",
)
revisited = [c for c in s.claims if len(c.evidence) >= 2]
check(
    len(revisited) >= 1,
    "at least one topic revisited with a second, independent probe",
    ", ".join(f"day {c.day} {c.concept} ({len(c.evidence)} probes)" for c in revisited[:3])
    or "none",
)

# ---------------------------------------------------------------------------
print(f"\n{BOLD}Requirement 5 — structured feedback at the end{OFF}")
fb = (final or {}).get("feedback") or {}
check(
    isinstance(fb, dict) and set(fb) >= {"summary", "strengths", "gaps", "next"},
    "feedback carries the four spec fields",
    f"keys present: {sorted(fb)}",
)
check(
    isinstance(fb.get("summary"), str) and len(fb.get("summary", "")) > 40,
    "summary is substantive prose",
    (fb.get("summary", "")[:160] + "…") if fb.get("summary") else "EMPTY",
)
for field in ("strengths", "gaps", "next"):
    val = fb.get(field)
    check(
        isinstance(val, list) and len(val) >= 1 and all(isinstance(x, str) and x for x in val),
        f"{field}: non-empty list of strings",
        " | ".join(str(x)[:80] for x in (val or [])[:3]) or "EMPTY",
    )

# ---------------------------------------------------------------------------
print(f"\n{BOLD}Requirement 1 — conversational, not a questionnaire{OFF}")
GENERIC = ("great question", "excellent!", "that's a great point", "well done!")
lower = [r.lower() for r in replies]
check(
    not any(g in r for r in lower for g in GENERIC),
    "no empty praise filler",
    "checked " + ", ".join(repr(g) for g in GENERIC),
)
check(
    all("*" not in r and "#" not in r and "\n-" not in r for r in replies),
    "replies are plain speakable prose (voice-ready)",
    "no markdown, bullets or code fences in any reply",
)
long_replies = [r for r in replies if len(r.split(". ")) > 5]
check(
    not long_replies,
    "replies stay short enough to be spoken",
    f"longest reply {max(len(r.split()) for r in replies)} words",
)
# A reaction means the reply does more than pose a question.
reacting = [r for r in replies[1:-1] if r and not r.strip().startswith(("What", "How", "Walk", "Tell"))]
check(
    len(reacting) >= max(1, (len(replies) - 2) // 2),
    "replies engage with the answer before asking",
    f"{len(reacting)} of {len(replies) - 2} mid-interview replies open with a reaction",
)

# ---------------------------------------------------------------------------
passed = sum(1 for ok, _, _ in results if ok)
total = len(results)
print(f"\n{BOLD}{passed}/{total} checks passed{OFF}")
if passed != total:
    print(f"{RED}Failing:{OFF}")
    for ok, name, _ in results:
        if not ok:
            print(f"  - {name}")
sys.exit(0 if passed == total else 1)
