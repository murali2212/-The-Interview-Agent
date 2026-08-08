"""The deterministic engine.

Not a toy mode. This is what runs when there is no key, no network, or a rate
limit at the worst possible moment, and it is why the demo cannot die during
judging. It scores answers from real textual features rather than pretending
to be a language model.

The feature that matters most is the bluff detector. Fluent, confident prose
that asserts ownership and scale while naming no mechanism, no number and no
trade-off scores BELOW an honest "I don't know" — because it should.
"""

from __future__ import annotations

import hashlib
import re

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

HEDGES = (
    "i think", "probably", "not sure", "i guess", "maybe", "might be", "i believe",
    "kind of", "sort of", "i don't know", "i dont know", "no idea", "never used",
    "never implemented", "i'd be guessing", "id be guessing", "can't remember",
)

TRADEOFFS = (
    "trade-off", "tradeoff", "trade off", " versus ", " vs ", "at the cost of",
    "downside", "the cost is", "you give up", "but that means", "in exchange",
    "at the expense of", "we accepted", "the catch is", "slower but", "faster but",
)

MECHANISMS = (
    "because", "which means", "so that", "under the hood", "the reason",
    "what happens is", "that causes", "leads to", "results in", "otherwise",
    "if you don't", "when it fails", "the way it works",
)

GROUNDING = (
    "i built", "we built", "i shipped", "we shipped", "in my project", "in our system",
    "i had to", "we ended up", "i wrote", "we ran", "i debugged", "my implementation",
    "our pipeline", "i measured", "we measured", "i tried", "we found",
)

#: Ownership and scale assertions. Harmless alone, damning with no substance.
ASSERTIONS = (
    "end to end", "end-to-end", "in production", "at scale", "a lot of",
    "one of my strongest", "heavily", "multiple times", "the whole pipeline",
    "handled all", "worked really well", "deployed it", "extensively",
    "from scratch", "fully optimized", "production ready", "production-ready",
)

#: Populated from the curriculum's own `tools` lists at import time by
#: `register_tools`, so "concrete" means concrete *to this cohort*.
_TOOL_TERMS: set[str] = set()

_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:ms|s|%|k|m|gb|mb|x|tokens?|qps|rps)?\b", re.I)


def register_tools(tools: list[str]) -> None:
    for t in tools:
        t = t.strip().lower()
        if len(t) >= 3:
            _TOOL_TERMS.add(t)


def _count(hay: str, terms) -> int:
    return sum(1 for t in terms if t in hay)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def features(text: str) -> dict[str, float]:
    raw = (text or "").strip()
    hay = f" {raw.lower()} "
    words = len([w for w in raw.split() if w])
    return {
        "words": words,
        "hedges": _count(hay, HEDGES),
        "tradeoffs": _count(hay, TRADEOFFS),
        "mechanisms": _count(hay, MECHANISMS),
        "grounding": _count(hay, GROUNDING),
        "assertive": _count(hay, ASSERTIONS),
        "tools": _count(hay, _TOOL_TERMS),
        "numbers": len(_NUMBER.findall(raw)),
        "length": clamp01((words - 8) / 55),
    }


def score_answer(text: str) -> dict[str, float | str | None]:
    f = features(text)

    if f["words"] == 0:
        return {
            "correctness": 0.0, "depth": 0.0, "specificity": 0.0, "signal": -0.9,
            "took_bait": None, "note": "No answer given.",
        }

    hedge_pen = min(0.45, f["hedges"] * 0.13)
    substance = f["mechanisms"] + f["numbers"] + f["tradeoffs"]
    bluff = min(0.40, 0.13 + min(3, f["assertive"]) * 0.09) if f["assertive"] and not substance else 0.0

    specificity = clamp01(
        0.08 + min(4, f["tools"]) * 0.11 + min(3, f["numbers"]) * 0.09
        + min(2, f["grounding"]) * 0.13 + f["length"] * 0.24 - hedge_pen - bluff * 0.7
    )
    depth = clamp01(
        0.08 + min(3, f["mechanisms"]) * 0.11 + min(2, f["tradeoffs"]) * 0.16
        + min(4, f["tools"]) * 0.05 + f["length"] * 0.30 - hedge_pen - bluff * 1.15
    )
    correctness = clamp01(
        0.14 + 0.42 * specificity + 0.34 * depth + 0.22 * f["length"]
        + min(4, f["tools"]) * 0.03 - hedge_pen * 0.8 - bluff * 0.9
    )

    composite = correctness * 0.5 + depth * 0.3 + specificity * 0.2
    signal = max(-1.0, min(1.0, composite * 2 - 1))

    # An honest admission is weak, not dishonest. Soften it — and make sure it
    # still lands above a confident bluff.
    honest = f["hedges"] >= 1 and f["words"] < 45 and not f["assertive"]
    if honest:
        signal = max(signal, -0.35)

    return {
        "correctness": round(correctness, 2),
        "depth": round(depth, 2),
        "specificity": round(specificity, 2),
        "signal": round(signal, 2),
        "took_bait": None,
        "note": _note(f, correctness, bluff, honest),
    }


def _note(f: dict[str, float], correctness: float, bluff: float, honest: bool) -> str:
    if bluff:
        return (
            "Claims ownership and scale but supplies no mechanism, number or trade-off to "
            "back it. Confident phrasing, no evidence underneath."
        )
    if honest:
        return "Openly admits not knowing this rather than bluffing — weak on content, straight about it."
    if correctness >= 0.7 and f["mechanisms"]:
        return (
            "Explains the mechanism rather than restating the definition"
            + (", and grounds it in their own build." if f["grounding"] else ".")
        )
    if correctness >= 0.45:
        return "Right shape, thin detail. The outline holds up but the specifics are missing."
    if f["words"] < 20:
        return "Too short to demonstrate understanding."
    return "States what it does but not why it behaves that way; nothing measured, nothing named."


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, tuple[str, ...]] = {
    "opening": (
        "Let's start somewhere you spent real time. Walk me through what you built for {concept}.",
        "Before we go deep, tell me about your work on {concept}. What did you actually put together?",
    ),
    "probe": (
        "When you worked on {concept}, what was the part that behaved differently from how you expected?",
        "Talk me through how {concept} actually works in what you built, not what it is for.",
    ),
    "follow-up": (
        "You touched on that. Push it one level further for me: why did {concept} behave that way?",
        "Stay on that for a second. What would have broken if you had done {concept} differently?",
    ),
    "verification": (
        "You should know this one well. What specifically did you have to configure for {concept}, and what went wrong first?",
        "Give me a detail from your own build of {concept} that someone who only read about it would not know.",
    ),
    "scenario": (
        "Suppose {concept} is silently returning poor results in production, but nothing is erroring. Where do you look first?",
        "Your {concept} works locally and fails under real traffic. Walk me through how you would find the cause.",
    ),
    "misconception": (
        "Since {concept} handles that automatically, you do not really need to measure it afterwards, right?",
        "Getting {concept} working basically removes that class of problem from the system, doesn't it?",
    ),
    "recovery": (
        "Let's step back a level. In your own words, what problem does {concept} solve?",
        "Simpler question. Why would a team bother with {concept} at all?",
    ),
    "closing": (
        "Last one. What was the hardest thing you hit in the cohort, and what would you do differently?",
        "To finish: which part of what you built are you least confident would survive real users?",
    ),
}


def question(kind: str, concept: str, seed: str = "") -> str:
    bank = _TEMPLATES.get(kind, _TEMPLATES["probe"])
    h = int(hashlib.sha1(f"{kind}|{concept}|{seed}".encode()).hexdigest()[:8], 16)
    return bank[h % len(bank)].format(concept=concept)


def summarise(facts: str) -> str:
    return facts.strip()[:600]
