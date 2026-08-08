"""The Director — orchestration and the single entry point.

`turn()` is the only function the outside world calls. The chat API calls it.
The voice agent will call the same function with a transcript string instead of
a typed one, which is why it returns plain speakable prose and holds all its
state under the caller's `sessionId`.

The interview is a search, not a script:

    Profile Analyst  telemetry            -> explicit priors
    Ledger           what to ask about    -> highest information gain
    Director         how to ask it        -> question kind
    Interviewer      the actual sentence  -> speakable question
    Assessor         the answer           -> signal back into the ledger
    Feedback         the whole thing      -> summary / strengths / gaps / next
"""

from __future__ import annotations

import logging

from . import config, knowledge, llm, offline
from .knowledge import day_context, days_mentioned
from .ledger import (
    Claim,
    ClaimSource,
    ClaimStatus,
    Evidence,
    apply_evidence,
    expected_information_gain,
    select_target,
    summarise,
)
from .models import Candidate, Feedback, InterviewRequest, InterviewResponse
from .normalize import Curriculum, load_curriculum
from .profile import analyse, topic_of
from .prompts.templates import (
    ASSESSOR_PROMPT,
    FEEDBACK_PROMPT,
    KIND_INSTRUCTIONS,
    QUESTION_PROMPT,
    SUMMARY_PROMPT,
)
from .normalize import build_profile
from . import session as store
from .session import Session, Turn

log = logging.getLogger("abtalks.director")

MAX_VERIFICATIONS = 3
MAX_RECOVERIES = 2

#: Follow-ups deepen a topic but spend a turn without opening a curriculum day.
#: In an eight-question interview that has to reach five days, at most three
#: turns can afford to revisit.
MAX_FOLLOWUPS = 3

_CURRICULUM: Curriculum | None = None


def curriculum() -> Curriculum:
    global _CURRICULUM
    if _CURRICULUM is None:
        _CURRICULUM = load_curriculum()
        tools: list[str] = []
        for d in _CURRICULUM.days.values():
            tools.extend(d.tools)
        offline.register_tools(tools)
    return _CURRICULUM


# ---------------------------------------------------------------------------
# Question kind policy
# ---------------------------------------------------------------------------


def _kind_count(s: Session, kind: str) -> int:
    return sum(1 for t in s.turns if t.kind == kind)


def _consecutive_struggles(s: Session) -> int:
    n = 0
    for t in reversed(s.answered):
        if (t.scores or {}).get("signal", 0) < -0.15:
            n += 1
        else:
            break
    return n


def _coverage_debt(s: Session) -> tuple[int, int]:
    """(questions still available, curriculum days still owed)."""
    remaining = config.MAX_QUESTIONS - len(s.answered)
    owed = config.MIN_DISTINCT_DAYS - len(s.days_covered)
    return remaining, owed


def _coverage_tight(s: Session) -> bool:
    """True when every remaining turn must open a NEW day to meet the floor.

    With a fixed eight-question budget this is the constraint that actually
    binds. Follow-ups, recovery questions and the closing flourish all spend a
    turn on a day already visited, and any one of them can quietly push the
    interview below the required coverage. When the budget is tight they lose.
    """
    remaining, owed = _coverage_debt(s)
    return owed >= remaining > 0


def decide_kind(s: Session, target: Claim | None) -> tuple[str, str]:
    asked = len(s.turns)
    prev = s.turns[-1] if s.turns else None
    last = (s.answered[-1].scores or {}) if s.answered else {}

    if asked == 0:
        return "opening", "Opening on ground the candidate should own."

    # Coverage outranks everything below, including the closing question.
    # Breadth is graded; a reflective sign-off is not.
    if _coverage_tight(s):
        remaining, owed = _coverage_debt(s)
        return (
            "probe",
            f"Coverage budget: {remaining} question(s) left and {owed} curriculum day(s) "
            f"still owed, so this turn must open new ground.",
        )

    if asked >= config.MAX_QUESTIONS - 1:
        return "closing", "Final question — inviting reflection."

    # Dignity rule. Capped, and never back to back: an interview that is all
    # recovery has stopped assessing and started consoling.
    if (
        _consecutive_struggles(s) >= 2
        and prev
        and prev.kind != "recovery"
        and _kind_count(s, "recovery") < MAX_RECOVERIES
    ):
        return "recovery", "Two weak answers in a row — stepping down a level before concluding."

    followups_left = _kind_count(s, "follow-up") < MAX_FOLLOWUPS

    # One answer is not a verdict. Go back for a second read on the same claim.
    if followups_left and prev and prev.kind != "follow-up":
        pc = s.claim_for(prev.day)
        if pc and pc.probes == 1 and pc.stakes >= 0.6 and not pc.corroborated:
            return (
                "follow-up",
                f"{pc.concept} is carrying a high-stakes verdict on one answer — second pass.",
            )

    # Strong and specific -> drill once.
    if (
        followups_left
        and last.get("signal", 0) > 0.2
        and last.get("specificity", 0) > 0.45
        and prev
        and prev.kind != "follow-up"
    ):
        return "follow-up", "That answer was specific and correct — pushing one level deeper."

    if (
        target
        and target.source is ClaimSource.ROLE_IMPLIED
        and target.probes == 0
        and _kind_count(s, "verification") < MAX_VERIFICATIONS
    ):
        return "verification", "Their role implies they own this and it has never been tested."

    if target and 0.35 < target.belief < 0.75 and _kind_count(s, "misconception") < 2:
        return "misconception", "Belief is mid-range — a wrong premise separates recall from understanding."

    if asked >= config.MAX_QUESTIONS // 2 and _kind_count(s, "scenario") < 2:
        return "scenario", "Moving from recall to applied judgement."

    return "probe", "Resolving the highest-value open uncertainty."


# ---------------------------------------------------------------------------
# Targeting
# ---------------------------------------------------------------------------


def _module_spread_exclusions(s: Session, covered: list[int]) -> tuple[set[int], int]:
    """Days to hold back so the interview spans the curriculum.

    Role-implied claims carry both a stakes boost and an information-gain
    bonus, which is correct in isolation and disastrous in aggregate: they
    cluster in one module and crowd out everything else. Until the interview
    has touched `MIN_DISTINCT_MODULES` modules, claims from already-visited
    modules step aside.
    """
    cur = curriculum()
    covered_modules = {m.n for d in covered if (m := cur.module_for(d))}
    if len(covered_modules) >= config.MIN_DISTINCT_MODULES:
        return set(), len(covered_modules)

    outside = [
        c for c in s.claims if (m := cur.module_for(c.day)) and m.n not in covered_modules
    ]
    if not outside:
        return set(), len(covered_modules)

    return (
        {c.day for c in s.claims if (m := cur.module_for(c.day)) and m.n in covered_modules},
        len(covered_modules),
    )


def pick_target(s: Session):
    covered = s.days_covered
    held_back, module_count = _module_spread_exclusions(s, covered)

    # Verify the boast.
    #
    # Information gain will never probe a role-implied claim sitting at 0.8 —
    # entropy there is nearly zero, so the maths calls it settled. But an
    # unverified claim is the most expensive thing to leave in a report, so it
    # gets an explicit rule rather than relying on the arithmetic.
    boast = [
        c
        for c in s.claims
        if c.source is ClaimSource.ROLE_IMPLIED
        and c.probes == 0
        and c.belief >= 0.7
        and c.day not in held_back
    ]
    if boast and len(s.turns) >= 1 and _kind_count(s, "verification") < MAX_VERIFICATIONS:
        pick = max(boast, key=lambda c: c.belief * c.stakes)
        return pick, (
            f"{pick.concept} (day {pick.day}) is implied by their role at "
            f"{pick.belief:.0%} confidence and has never been tested."
        )

    target = select_target(
        s.claims,
        days_covered=covered,
        questions_asked=len(s.answered),
        min_questions=config.MIN_QUESTIONS,
        min_distinct_days=config.MIN_DISTINCT_DAYS,
        exclude_days=held_back,
    )

    if target.claim is None and held_back:
        # Spread was too strict for what is left on the board — drop it rather
        # than end the interview early.
        target = select_target(
            s.claims,
            days_covered=covered,
            questions_asked=len(s.answered),
            min_questions=config.MIN_QUESTIONS,
            min_distinct_days=config.MIN_DISTINCT_DAYS,
        )
    elif held_back and target.claim is not None:
        module = curriculum().module_for(target.claim.day)
        if module:
            return target.claim, (
                f"{target.reason} Module spread: {module_count} of "
                f"{config.MIN_DISTINCT_MODULES} required modules touched so far, so this pick "
                f"opens module {module.n} ({module.title})."
            )

    return target.claim, target.reason


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------


def _verdict_line(scores: dict) -> str:
    """Turn the numbers into an instruction the interviewer can act on.

    Without this the model has the transcript but no opinion about it, so it
    politely moves on regardless of whether the answer was excellent or empty —
    which is exactly what makes an AI interviewer feel like a questionnaire.
    """
    correctness = float(scores.get("correctness") or 0.0)
    depth = float(scores.get("depth") or 0.0)
    specificity = float(scores.get("specificity") or 0.0)
    note = str(scores.get("note") or "").strip()

    if scores.get("took_bait") is True:
        verdict = (
            "They accepted a false premise you planted. Correct it directly but without "
            "point-scoring, then continue."
        )
    elif correctness >= 0.7 and specificity >= 0.55:
        verdict = (
            "Strong and specific. Name the exact detail that convinced you, then go deeper "
            "rather than re-testing it."
        )
    elif correctness >= 0.7:
        verdict = "Correct but generic. Acknowledge it, then press for something from their own build."
    elif depth < 0.35 and correctness >= 0.4:
        verdict = (
            "They described what it does, not why. Say that plainly and ask for the mechanism."
        )
    elif correctness >= 0.4:
        verdict = "Partly there, detail missing. Push on the specific part they skipped."
    elif "honest" in note.lower() or "admits" in note.lower():
        verdict = "An honest admission. Respect it, do not labour it, move to firmer ground."
    else:
        verdict = "Weak. Do not pretend otherwise — say what was missing, then step down a level."

    return f"{verdict} (your read: {note})" if note else verdict


def _recent_block(s: Session, n: int = 2) -> str:
    tail = [t for t in s.answered][-n:]
    if not tail:
        return "NO PREVIOUS ANSWER YET — there is nothing to react to."

    lines = ["WHAT HAS JUST BEEN SAID"]
    if s.summary:
        lines.append(f"  earlier in this interview: {s.summary}")

    for t in tail[:-1]:
        lines.append(f"  you asked: {t.question}")
        lines.append(f"  they said: {(t.answer or '')[:400]}")

    last = tail[-1]
    lines.append("")
    lines.append("THEIR MOST RECENT ANSWER — react to THIS before you ask anything")
    lines.append(f"  you asked: {last.question}")
    lines.append(f"  they said: {(last.answer or '')[:900]}")
    if last.scores:
        lines.append(f"  how it landed: {_verdict_line(last.scores)}")
    return "\n".join(lines)


def compose_question(s: Session, claim: Claim, kind: str, reason: str) -> str:
    ctx = day_context(curriculum(), claim.day)
    fallback = offline.question(kind, claim.concept, seed=f"{s.id}:{len(s.turns)}")

    text = llm.ask(
        QUESTION_PROMPT,
        {
            "day": claim.day,
            "concept": claim.concept,
            "belief": claim.belief,
            "prior": claim.prior,
            "why_prior": claim.why_prior or "no prior signal",
            "reason": reason,
            "kind": kind,
            "kind_instruction": KIND_INSTRUCTIONS.get(kind, KIND_INSTRUCTIONS["probe"]),
            "recent": _recent_block(s),
            "history": [],
            **ctx,
        },
        tier=llm.TIER_MAIN,
        fallback=fallback,
    )
    spoken = llm.speakable(text)
    return spoken or fallback


# ---------------------------------------------------------------------------
# Assessing
# ---------------------------------------------------------------------------


def assess(s: Session, turn: Turn, answer: str) -> dict:
    baseline = offline.score_answer(answer)
    ctx = day_context(curriculum(), turn.day)

    got = llm.ask_json(
        ASSESSOR_PROMPT,
        {
            "kind": turn.kind,
            "concept": turn.concept,
            "day": turn.day,
            "day_title": ctx["day_title"],
            "objectives": ctx["objectives"],
            "question": turn.question,
            "answer": answer[:4000],
            "baseline": (
                f"correctness {baseline['correctness']}, depth {baseline['depth']}, "
                f"specificity {baseline['specificity']}"
            ),
        },
        tier=llm.TIER_MAIN,
        fallback={},
    )

    def num(key: str) -> float:
        v = got.get(key)
        if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0:
            return round(float(v), 2)
        return float(baseline[key])  # type: ignore[arg-type]

    correctness, depth, specificity = num("correctness"), num("depth"), num("specificity")
    signal = round(max(-1.0, min(1.0, (0.5 * correctness + 0.3 * depth + 0.2 * specificity) * 2 - 1)), 2)

    took_bait = got.get("took_bait")
    if turn.kind != "misconception" or not isinstance(took_bait, bool):
        took_bait = None
    if took_bait is True:
        signal = min(signal, -0.4)  # accepted a false premise

    note = got.get("note")
    if not isinstance(note, str) or len(note.strip()) < 12:
        note = str(baseline["note"])

    claims = [c for c in got.get("claims", []) if isinstance(c, str) and c.strip()][:3]

    return {
        "correctness": correctness,
        "depth": depth,
        "specificity": specificity,
        "signal": signal,
        "took_bait": took_bait,
        "note": note.strip(),
        "claims": claims,
    }


# ---------------------------------------------------------------------------
# Turn machine
# ---------------------------------------------------------------------------


def _ask_next(s: Session) -> str:
    claim, reason = pick_target(s)
    if claim is None:
        claim = max(s.claims, key=expected_information_gain)
        reason = "Falling back to the least-resolved claim."

    kind, why = decide_kind(s, claim)

    # A follow-up that changes subject is not a follow-up. Re-point it at the
    # claim we were just testing — unless coverage is tight, in which case
    # `decide_kind` has already ruled follow-ups out and this must not fire.
    derived_from = None
    if kind == "follow-up" and s.answered and not _coverage_tight(s):
        prev = s.answered[-1]
        pc = s.claim_for(prev.day)
        if pc is not None and pc.probes < 2:
            claim, reason = pc, why
        derived_from = prev.n

    s.log("Director", "target-selected", reason)
    text = compose_question(s, claim, kind, reason)

    turn = Turn(
        n=len(s.turns) + 1,
        kind=kind,
        day=claim.day,
        concept=claim.concept,
        question=text,
        rationale=why,
        derived_from=derived_from,
    )
    s.turns.append(turn)
    s.log("Interviewer", f"question:{kind}", text)
    return text


def _should_conclude(s: Session) -> tuple[bool, str]:
    answered = len(s.answered)
    days = len(s.days_covered)

    if answered >= config.MAX_QUESTIONS:
        return True, f"Reached the question ceiling ({config.MAX_QUESTIONS})."
    if answered < config.MIN_QUESTIONS:
        return False, f"{answered} of {config.MIN_QUESTIONS} required questions answered."
    if days < config.MIN_DISTINCT_DAYS:
        return False, f"Covered {days} of {config.MIN_DISTINCT_DAYS} required days."

    unresolved = [c for c in s.claims if c.probes == 1 and c.stakes >= 0.6 and not c.corroborated]
    if unresolved:
        return False, f"{len(unresolved)} high-stakes claim(s) still rest on one answer."
    return True, "Coverage met and no high-stakes uncertainty remains open."


def _update_summary(s: Session) -> None:
    resolved = [c for c in s.claims if c.status in (ClaimStatus.SUPPORTED, ClaimStatus.REFUTED)]
    facts = (
        f"{len(s.answered)} questions answered on days {s.days_covered}. "
        + "; ".join(
            f"{c.concept}={'holds' if c.belief >= 0.5 else 'does not hold'}" for c in resolved[:6]
        )
    )
    s.summary = llm.ask(SUMMARY_PROMPT, {"facts": facts}, tier=llm.TIER_FAST, fallback=facts)


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


def calibration_rows(s: Session) -> list[dict]:
    rows = []
    for c in sorted(s.claims, key=lambda x: -abs(x.delta)):
        if not c.evidence:
            continue
        if c.prior >= 0.55 and c.delta <= -0.15:
            verdict = "overclaimed"
        elif c.prior <= 0.45 and c.delta >= 0.15:
            verdict = "underrated"
        elif c.belief >= 0.65:
            verdict = "confirmed-strength"
        elif c.belief <= 0.35:
            verdict = "confirmed-gap"
        else:
            verdict = "partially-supported"
        rows.append(
            {
                "day": c.day,
                "concept": c.concept,
                "prior": round(c.prior, 2),
                "posterior": round(c.belief, 2),
                "delta": round(c.delta, 2),
                "verdict": verdict,
                "source": c.source.value,
            }
        )
    return rows


def build_feedback(s: Session) -> Feedback:
    rows = calibration_rows(s)
    cal = "\n".join(
        f"  day {r['day']} {r['concept']}: believed {r['prior']:.2f} -> found {r['posterior']:.2f} ({r['verdict']})"
        for r in rows
    ) or "  (no claims were resolved)"

    highlights = "\n".join(
        f"  Q{t.n} [{t.kind}] {t.question}\n     answer: {(t.answer or '')[:280]}\n     assessed: {t.scores.get('note')}"
        for t in s.answered
    )

    strong = [r for r in rows if r["verdict"] in ("confirmed-strength", "underrated")]
    weak = [r for r in rows if r["verdict"] in ("confirmed-gap", "overclaimed")]

    # Fed to the model as a closed list. Without this it happily writes
    # "claimed to have built the whole pipeline, indicating experience" — which
    # turns the exact assertion that failed verification into a strength.
    confirmed_block = (
        "\n".join(
            f"  {r['concept']} (day {r['day']}) — belief rose to {r['posterior']:.2f}" for r in strong
        )
        or "  (none — nothing was confirmed under questioning)"
    )
    unsupported_block = (
        "\n".join(
            f"  {r['concept']} (day {r['day']}) — believed {r['prior']:.2f}, fell to "
            f"{r['posterior']:.2f} when probed"
            for r in weak
        )
        or "  (none)"
    )

    fallback = {
        "summary": (
            f"We covered {len(s.answered)} questions across cohort days "
            f"{', '.join(map(str, s.days_covered))}. "
            + (f"You held up on {strong[0]['concept']}. " if strong else "")
            + (f"The weakest area was {weak[0]['concept']}." if weak else "")
        ),
        "strengths": [f"{r['concept']} (day {r['day']}) held up under questioning." for r in strong[:3]]
        or [
            "Nothing was confirmed under questioning in this session — the areas we covered did "
            "not hold up when probed for mechanism."
        ],
        "gaps": [f"{r['concept']} (day {r['day']}) did not hold up when probed." for r in weak[:3]]
        or ["No decisive gap emerged in the areas we covered."],
        "next": [
            f"Revisit cohort day {r['day']} ({r['concept']}) and write up the trade-off you chose."
            for r in weak[:3]
        ]
        or ["Extend the capstone with a failure mode you have not handled yet."],
    }

    got = llm.ask_json(
        FEEDBACK_PROMPT,
        {
            "name": s.profile.name,
            "role": s.profile.role or "unspecified role",
            "years": s.profile.years,
            "calibration": cal,
            "confirmed": confirmed_block,
            "unsupported": unsupported_block,
            "highlights": highlights,
            "questions": len(s.answered),
            "days": ", ".join(map(str, s.days_covered)),
        },
        tier=llm.TIER_MAIN,
        fallback=fallback,
    )

    def arr(key: str) -> list[str]:
        v = got.get(key)
        items = [x.strip() for x in v if isinstance(x, str) and x.strip()] if isinstance(v, list) else []
        return items[:4] or fallback[key]  # type: ignore[return-value]

    summary = got.get("summary")
    if not isinstance(summary, str) or len(summary.strip()) < 30:
        summary = fallback["summary"]

    return Feedback(
        summary=summary.strip(),
        strengths=arr("strengths"),
        gaps=arr("gaps"),
        next=arr("next"),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _insight(s: Session) -> dict:
    return {
        "sessionId": s.id,
        "candidate": s.headline,
        "questionsAsked": len(s.answered),
        "daysCovered": s.days_covered,
        "patterns": s.patterns,
        "ledger": [
            {
                "day": c.day,
                "concept": c.concept,
                "source": c.source.value,
                "prior": round(c.prior, 2),
                "belief": round(c.belief, 2),
                "stakes": round(c.stakes, 2),
                "status": c.status.value,
                "probes": c.probes,
            }
            for c in sorted(s.claims, key=lambda x: -x.stakes)
        ],
        "calibration": calibration_rows(s),
        "summary": summarise(s.claims),
        "transcript": [
            {
                "n": t.n,
                "kind": t.kind,
                "day": t.day,
                "concept": t.concept,
                "question": t.question,
                "rationale": t.rationale,
                "derivedFrom": t.derived_from,
                "answer": t.answer,
                "scores": t.scores,
            }
            for t in s.turns
        ],
        "trace": [
            {"seq": e.seq, "agent": e.agent, "action": e.action, "detail": e.detail}
            for e in s.events
        ],
        "engine": llm.describe(),
    }


def start(session_id: str, candidate: Candidate) -> Session:
    profile = build_profile(candidate)
    analysis = analyse(profile, curriculum())

    s = Session(
        id=session_id,
        candidate=candidate,
        profile=profile,
        claims=analysis.claims,
        patterns=analysis.patterns,
        headline=analysis.headline,
    )
    s.log("ProfileAnalyst", "priors-seeded", f"{len(analysis.claims)} beliefs formed before asking anything.")
    for p in analysis.patterns:
        s.log("ProfileAnalyst", "pattern", p)

    store.put(s)
    _ask_next(s)
    return s


def respond(s: Session, message: str) -> None:
    pending = s.pending
    if pending is None:
        _ask_next(s)
        return

    pending.answer = message
    pending.scores = assess(s, pending, message)
    s.log("Assessor", "scored", pending.scores["note"])

    claim = s.claim_for(pending.day)
    if claim is not None:
        before = claim.belief
        updated = apply_evidence(
            claim,
            Evidence(
                turn=pending.n,
                signal=pending.scores["signal"],
                weight=0.5 + 0.5 * pending.scores["depth"],
                note=pending.scores["note"],
            ),
        )
        s.replace_claim(updated)
        s.log(
            "Ledger",
            "belief-updated",
            f"{updated.concept}: {before:.2f} -> {updated.belief:.2f} ({updated.status.value})",
        )

    # Anything they asserted becomes testable — mapped onto a real curriculum
    # day rather than invented from a sentence fragment.
    for assertion in pending.scores.get("claims", []):
        for day in days_mentioned(curriculum(), assertion):
            if s.claim_for(day):
                continue
            d = curriculum().day(day)
            if d is None:
                continue
            s.claims.append(
                Claim(
                    day=day,
                    concept=topic_of(d),
                    source=ClaimSource.INTERVIEW_ASSERTION,
                    prior=0.6,
                    belief=0.6,
                    stakes=curriculum().stakes(day),
                    why_prior=f'asserted mid-interview: "{assertion}"',
                )
            )
            s.log("ClaimVerifier", "registered", f'"{assertion}" -> day {day}, queued for verification.')

    if len(s.answered) % 3 == 0:
        _update_summary(s)

    done, why = _should_conclude(s)
    if done:
        s.log("Director", "concluded", why)
        s.feedback = build_feedback(s)
        s.done = True
        s.log("Feedback", "issued", s.feedback.summary[:160])
    else:
        _ask_next(s)


def turn(req: InterviewRequest) -> InterviewResponse:
    """Single entry point. Chat calls this; voice will call the same function."""
    if req.is_start():
        store.drop(req.sessionId)
        s = start(req.sessionId, req.candidate)  # type: ignore[arg-type]
        opening = (
            f"Hello {s.profile.name.split()[0] if s.profile.name else 'there'}, thanks for making "
            f"the time. I have looked over your cohort work and I would like to dig into a few "
            f"parts of it."
        )
        return InterviewResponse(
            reply=f"{opening} {s.turns[-1].question}",
            done=False,
            insight=_insight(s),
        )

    s = store.get(req.sessionId)
    if s is None:
        return InterviewResponse(
            reply=(
                "I do not have an interview open under that session. Send the candidate profile "
                "to start a new one."
            ),
            done=False,
        )

    if s.done:
        return InterviewResponse(
            reply="That interview is already complete.",
            done=True,
            feedback=s.feedback,
            insight=_insight(s),
        )

    respond(s, req.message or "")
    store.put(s)

    if s.done:
        return InterviewResponse(
            reply="That is everything I wanted to cover. Thank you for walking me through your work.",
            done=True,
            feedback=s.feedback,
            insight=_insight(s),
        )

    return InterviewResponse(reply=s.turns[-1].question, done=False, insight=_insight(s))
