"""Drive a real interview against the live model and print it.

    python scripts/live_check.py                 # Harold, bluffing
    python scripts/live_check.py CAND-003 strong

Personas differ in the FEATURES their answers carry, which is exactly what the
assessor scores on, so they produce genuinely different reports.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm, session as store  # noqa: E402
from app.director import turn  # noqa: E402
from app.models import InterviewRequest  # noqa: E402
from app.normalize import load_candidates  # noqa: E402

ANSWERS = {
    "strong": [
        "Our retrieval looked fine but answers were wrong, so I measured the stages separately: "
        "context recall was 0.62 while faithfulness was 0.91. The generator was fine and retrieval "
        "was starving it. Raising chunk overlap from zero to eighty tokens fixed most of it, at "
        "about fifteen percent more storage in Chroma.",
        "I set the router to check for numeric intent first. If the question mentions an amount or "
        "a date range it goes to SQLite, otherwise it goes to vector search. The failure case was "
        "'how much did I pay last year' matching a policy document semantically, which is why the "
        "rule fires before the embedding does.",
        "We logged every tool call with its arguments and latency. The thing that caught us was "
        "the agent retrying a failed call forever, so I added a step budget of six and a hard "
        "timeout of thirty seconds per tool.",
        "I validated the model output with a Pydantic schema and retried once on a parse failure. "
        "About four percent of calls needed the retry. Anything that failed twice returned a "
        "fallback instead of guessing.",
    ],
    "bluff": [
        "Yes, I've done a lot of that. I built the whole pipeline end to end in production and "
        "handled all the scaling, and it worked really well.",
        "That's one of my strongest areas. I deployed it multiple times and optimized it heavily "
        "for production workloads.",
        "We had that in our stack from the start. It worked well at scale, no real issues.",
        "I've worked with that extensively across several systems. It's fairly standard once you "
        "have the architecture right.",
    ],
    "honest": [
        "I don't know that one well. I read about it but never implemented it, so I'd be guessing.",
        "Honestly that part I skipped. I know roughly what it's for but not how it works.",
        "I'm not sure. I think it's about retrieval quality but I might be mixing it up.",
        "That one I can't answer properly. I never got to it in the cohort.",
    ],
}

C = {
    "dim": lambda s: f"\x1b[2m{s}\x1b[0m",
    "b": lambda s: f"\x1b[1m{s}\x1b[0m",
    "cy": lambda s: f"\x1b[36m{s}\x1b[0m",
    "gr": lambda s: f"\x1b[32m{s}\x1b[0m",
    "rd": lambda s: f"\x1b[31m{s}\x1b[0m",
    "ye": lambda s: f"\x1b[33m{s}\x1b[0m",
}


def main() -> None:
    cid = sys.argv[1] if len(sys.argv) > 1 else "CAND-008"
    persona = sys.argv[2] if len(sys.argv) > 2 else "bluff"
    bank = ANSWERS.get(persona, ANSWERS["strong"])

    candidates = {c.member.id: c for c in load_candidates()}
    cand = candidates[cid]

    print(C["b"]("\n  ABTALKS INTERVIEW AGENT"))
    print(C["dim"](f"  engine: {llm.describe()}"))
    print(
        C["dim"](
            f"  candidate: {cand.member.name} — {cand.member.jobRole}, "
            f"{cand.member.yearsExperience}y · persona: {persona}\n"
        )
    )

    store.clear()
    sid = "live-check"
    res = turn(InterviewRequest(sessionId=sid, candidate=cand))

    n = 0
    while True:
        tr = (res.insight or {}).get("transcript", [])
        cur = tr[-1] if tr else None
        if cur:
            print(
                C["dim"](
                    f"  [{cur['kind']} · day {cur['day']} · {cur['concept']}"
                    + (f" · follows Q{cur['derivedFrom']}" if cur["derivedFrom"] else "")
                    + "]"
                )
            )
        print(f"  {C['b']('Q' + str(n + 1))}  {res.reply}")
        if cur and cur.get("rationale"):
            print(C["dim"](f"       why now: {cur['rationale']}"))

        if res.done:
            break
        answer = bank[n % len(bank)]
        print(C["cy"](f"  A   {answer[:160]}{'…' if len(answer) > 160 else ''}"))

        res = turn(InterviewRequest(sessionId=sid, message=answer))

        tr = (res.insight or {}).get("transcript", [])
        scored = [t for t in tr if t.get("scores")]
        if scored:
            s = scored[-1]["scores"]
            print(
                C["dim"](
                    f"       scored c={s['correctness']:.2f} d={s['depth']:.2f} "
                    f"s={s['specificity']:.2f} signal={s['signal']:+.2f}"
                )
            )
            print(C["dim"](f"       {s['note']}"))
        print()
        n += 1
        if n > 16:
            break

    fb = res.feedback
    ins = res.insight or {}
    if fb:
        print(C["b"]("\n  FEEDBACK"))
        print(f"  {fb.summary}\n")
        print(C["b"]("  Strengths"))
        for x in fb.strengths:
            print(C["gr"](f"   + {x}"))
        print(C["b"]("\n  Gaps"))
        for x in fb.gaps:
            print(C["rd"](f"   - {x}"))
        print(C["b"]("\n  Next"))
        for x in fb.next:
            print(C["ye"](f"   > {x}"))

    cal = ins.get("calibration", [])
    if cal:
        print(C["b"]("\n  CALIBRATION  ") + C["dim"]("believed before -> found"))
        for r in cal[:12]:
            arrow = C["gr"]("^") if r["delta"] > 0.02 else C["rd"]("v") if r["delta"] < -0.02 else "-"
            print(
                f"   day {r['day']:>2}  {r['concept'][:34]:<34} "
                f"{r['prior']:.2f} {arrow} {r['posterior']:.2f}  {C['dim'](r['verdict'])}"
            )

    print(
        C["dim"](
            f"\n  {ins.get('questionsAsked')} questions · days {ins.get('daysCovered')} · "
            f"ledger {ins.get('summary')}\n"
        )
    )


if __name__ == "__main__":
    main()
