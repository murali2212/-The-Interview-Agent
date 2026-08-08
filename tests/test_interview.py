"""Full-loop tests against the graded contract.

Run offline (deterministic engine) so they are fast and stable. The live Groq
path is exercised separately by scripts/live_check.py.
"""

from __future__ import annotations

import os

os.environ["OFFLINE_ONLY"] = "true"

import pytest  # noqa: E402

from app import config, session as store  # noqa: E402

config.OFFLINE_ONLY = True

from app.director import turn  # noqa: E402
from app.models import Candidate, InterviewRequest  # noqa: E402
from app.normalize import load_candidates  # noqa: E402

CANDIDATES = {c.member.id: c for c in load_candidates()}

STRONG = (
    "We hit this directly. Our retrieval was returning fluent but wrong answers, so I measured "
    "the stages separately: recall was about 0.62 while the generator was fine at 0.91. "
    "I raised chunk overlap from zero to eighty tokens because the boundary was cutting "
    "definitions in half, which fixed most of it. The trade-off was roughly fifteen percent "
    "more storage in Chroma."
)
BLUFF = (
    "Yes, I've done a lot of this. I built the whole pipeline end to end in production, "
    "handled all the scaling, and it worked really well at scale."
)
HONEST = "I don't know that one, I never implemented it myself so I'd be guessing."


def run(candidate: Candidate, answer, session_id="t1", limit=20):
    store.clear()
    res = turn(InterviewRequest(sessionId=session_id, candidate=candidate))
    replies = [res]
    i = 0
    while not res.done and i < limit:
        i += 1
        text = answer(i) if callable(answer) else answer
        res = turn(InterviewRequest(sessionId=session_id, message=text))
        replies.append(res)
    return res, replies


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_start_returns_reply_and_not_done():
    store.clear()
    res = turn(InterviewRequest(sessionId="s1", candidate=CANDIDATES["CAND-001"]))
    assert isinstance(res.reply, str) and len(res.reply) > 20
    assert res.done is False
    assert res.feedback is None


def test_unknown_session_does_not_crash():
    store.clear()
    res = turn(InterviewRequest(sessionId="nope", message="hello"))
    assert res.done is False
    assert "session" in res.reply.lower()


def test_final_turn_carries_the_required_feedback_shape():
    res, _ = run(CANDIDATES["CAND-001"], STRONG)
    assert res.done is True
    fb = res.feedback
    assert fb is not None
    assert isinstance(fb.summary, str) and len(fb.summary) > 30
    for field in (fb.strengths, fb.gaps, fb.next):
        assert isinstance(field, list) and field
        assert all(isinstance(x, str) and x.strip() for x in field)


def test_answering_after_completion_is_safe():
    res, _ = run(CANDIDATES["CAND-003"], STRONG, session_id="done1")
    again = turn(InterviewRequest(sessionId="done1", message="anything"))
    assert again.done is True
    assert again.feedback is not None


# ---------------------------------------------------------------------------
# Graded minimums
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cid", ["CAND-001", "CAND-008", "CAND-010", "CAND-011", "CAND-017"])
def test_rubric_floors_for_varied_profiles(cid):
    res, _ = run(CANDIDATES[cid], STRONG, session_id=f"r-{cid}")
    ins = res.insight
    assert res.done is True
    assert ins["questionsAsked"] >= config.RUBRIC_FLOOR_QUESTIONS
    assert len(ins["daysCovered"]) >= config.RUBRIC_FLOOR_DAYS


def test_replies_are_speakable():
    """No markdown survives to `reply`: the voice transport reads it verbatim."""
    _, replies = run(CANDIDATES["CAND-002"], STRONG)
    for r in replies:
        assert "**" not in r.reply
        assert "```" not in r.reply
        assert "\n-" not in r.reply
        assert len(r.reply) < 700


# ---------------------------------------------------------------------------
# Adaptivity
# ---------------------------------------------------------------------------


def test_more_than_one_question_kind_is_used():
    res, _ = run(CANDIDATES["CAND-005"], STRONG)
    kinds = {t["kind"] for t in res.insight["transcript"]}
    assert len(kinds) >= 3, kinds


def test_a_struggling_candidate_is_offered_recovery():
    res, _ = run(CANDIDATES["CAND-010"], HONEST)
    kinds = [t["kind"] for t in res.insight["transcript"]]
    assert "recovery" in kinds, kinds


def test_follow_ups_derive_from_an_earlier_turn():
    res, _ = run(CANDIDATES["CAND-003"], STRONG)
    derived = [t for t in res.insight["transcript"] if t["derivedFrom"]]
    assert derived, "no question referenced a previous turn"
    for t in derived:
        assert t["derivedFrom"] < t["n"]


def test_context_is_maintained_across_turns():
    res, _ = run(CANDIDATES["CAND-001"], STRONG)
    tr = res.insight["transcript"]
    answered = [t for t in tr if t["answer"]]
    assert len(answered) >= config.RUBRIC_FLOOR_QUESTIONS
    assert all(t["scores"] for t in answered), "an answered turn lost its assessment"


# ---------------------------------------------------------------------------
# Scoring behaviour — the thing the project is actually about
# ---------------------------------------------------------------------------


def test_a_bluff_scores_below_an_honest_admission():
    from app.offline import score_answer

    bluff = score_answer(BLUFF)
    honest = score_answer(HONEST)
    strong = score_answer(STRONG)

    assert strong["signal"] > honest["signal"] > bluff["signal"], (
        strong["signal"], honest["signal"], bluff["signal"]
    )
    assert "no mechanism" in str(bluff["note"]).lower()


def test_beliefs_move_and_stay_in_range():
    res, _ = run(CANDIDATES["CAND-001"], STRONG)
    for c in res.insight["ledger"]:
        assert 0.02 <= c["belief"] <= 0.98
        if c["probes"] == 0:
            assert c["belief"] == c["prior"]


def test_calibration_is_produced_and_well_formed():
    res, _ = run(CANDIDATES["CAND-005"], STRONG)
    cal = res.insight["calibration"]
    assert cal, "no calibration rows"
    valid = {"overclaimed", "underrated", "confirmed-strength", "confirmed-gap", "partially-supported"}
    for r in cal:
        assert r["verdict"] in valid
        assert 0 <= r["prior"] <= 1 and 0 <= r["posterior"] <= 1


def test_a_bluffing_senior_gets_marked_down():
    """Harold is a Distinguished Engineer of 28 years. If he answers with pure
    confidence and no substance, role-implied claims must fall."""
    res, _ = run(CANDIDATES["CAND-008"], BLUFF, session_id="harold")
    cal = res.insight["calibration"]
    dropped = [r for r in cal if r["delta"] < -0.1]
    assert dropped, "confident-but-empty answers did not move any belief down"


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_junk_input_does_not_crash_the_panel():
    junk = ["", "   ", "???", "idk", "<script>alert(1)</script>", "a" * 5000, "{}{{}}"]
    res, _ = run(CANDIDATES["CAND-016"], lambda i: junk[i % len(junk)], session_id="junk")
    assert res.done is True
    assert res.feedback is not None


def test_the_sparse_profile_still_clears_the_floors():
    """Mia completed 14 missions over 9 commit days and skipped five topics."""
    res, _ = run(CANDIDATES["CAND-011"], HONEST, session_id="sparse")
    assert res.insight["questionsAsked"] >= config.RUBRIC_FLOOR_QUESTIONS
    assert len(res.insight["daysCovered"]) >= config.RUBRIC_FLOOR_DAYS


def test_every_agent_reports_into_the_trace():
    res, _ = run(CANDIDATES["CAND-001"], STRONG)
    agents = {e["agent"] for e in res.insight["trace"]}
    for expected in ("ProfileAnalyst", "Director", "Interviewer", "Assessor", "Ledger", "Feedback"):
        assert expected in agents, f"{expected} never reported"
