"""Agentic RAG over the cohort record.

Pattern from Ebbelaar's ai-cookbook (`knowledge/agentic-rag`): instead of
embedding a corpus and hoping cosine similarity returns the right chunk, give
the model tools to search and read the sources directly, and let it decide what
to look at next.

That is the right call here, and not only because it is fashionable. The corpus
is 31 short, exhaustively-titled documents. "Day 8 is Vector Databases" is an
exact fact. Embedding it and retrieving approximately can only lose
information. Grep and read cannot.

Two sources are exposed, which is what makes this useful rather than decorative:

    the CURRICULUM  — what the cohort taught
    the RECORD      — what this candidate actually did with it

An interviewer that can cross-reference those two on its own initiative can
notice "they are talking about reranking, which is day 10, which they failed
twice" without anyone hard-coding that path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .normalize import Curriculum, MissionState, Profile

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "is",
    "it", "how", "what", "why", "when", "do", "does", "you", "your", "i", "we",
    "that", "this", "did", "was", "were", "be", "have", "has", "about", "my",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9@_.-]+", (text or "").lower()) if t not in _STOP and len(t) > 2]


@dataclass
class Hit:
    day: int
    title: str
    type: str
    score: float
    matched: list[str]


def search_curriculum(curriculum: Curriculum, query: str, limit: int = 5) -> list[Hit]:
    """Grep-style lexical search across titles, objectives and tools.

    Tool names are weighted highest: if a candidate says "Chroma", the day that
    lists Chroma in its tooling is not a probabilistic guess, it is the answer.
    """
    terms = _tokens(query)
    if not terms:
        return []

    hits: list[Hit] = []
    for day in curriculum.days.values():
        title = day.title.lower()
        tools = " ".join(day.tools).lower()
        objectives = " ".join(day.objectives).lower()

        score = 0.0
        matched: list[str] = []
        for t in terms:
            if t in tools:
                score += 3.0
                matched.append(t)
            elif t in title:
                score += 2.0
                matched.append(t)
            elif t in objectives:
                score += 1.0
                matched.append(t)
        if score:
            hits.append(
                Hit(day=day.day, title=day.title, type=day.type, score=score, matched=sorted(set(matched)))
            )

    hits.sort(key=lambda h: (-h.score, h.day))
    return hits[:limit]


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def read_day(curriculum: Curriculum, day: int) -> dict[str, Any]:
    d = curriculum.day(day)
    if d is None:
        return {"error": f"Day {day} is not in the curriculum (valid range 1-31)."}
    return {
        "day": d.day,
        "title": d.title,
        "type": d.type,
        "module": f"{d.module_n}. {d.module_title}",
        "objectives": list(d.objectives),
        "tools": list(d.tools),
    }


def list_days(curriculum: Curriculum, module: int | None = None) -> list[dict[str, Any]]:
    out = []
    for d in sorted(curriculum.days.values(), key=lambda x: x.day):
        if module is not None and d.module_n != module:
            continue
        out.append({"day": d.day, "title": d.title, "type": d.type, "module": d.module_n})
    return out


def candidate_record(profile: Profile, day: int) -> dict[str, Any]:
    """What this candidate actually did on a given day.

    Deliberately distinguishes 'not in the record' from 'skipped'. The mission
    list is a sample, and reporting an unlisted day as avoidance would be a
    fabrication.
    """
    state = profile.state_of(day)
    rec = profile.records.get(day)
    if state is MissionState.UNOBSERVED:
        return {
            "day": day,
            "state": "unobserved",
            "note": "Not present in the submitted mission sample. This is NOT evidence of avoidance.",
        }
    return {
        "day": day,
        "state": state.value,
        "attempts": rec.attempts if rec else None,
        "title_in_record": rec.title if rec else None,
    }


# ---------------------------------------------------------------------------
# Context assembly (the deterministic path the director always uses)
# ---------------------------------------------------------------------------


def day_context(curriculum: Curriculum, day: int) -> dict[str, str]:
    """Formatted block for the question prompt."""
    d = curriculum.day(day)
    if d is None:
        return {
            "day_title": f"Day {day}",
            "day_type": "BUILD",
            "module_title": "",
            "objectives": "  (not available)",
            "tools": "",
        }
    return {
        "day_title": d.title,
        "day_type": d.type,
        "module_title": f"{d.module_n}. {d.module_title}",
        "objectives": "\n".join(f"    - {o}" for o in d.objectives),
        "tools": ", ".join(d.tools),
    }


def days_mentioned(curriculum: Curriculum, text: str, limit: int = 3) -> list[int]:
    """Which curriculum days is this answer actually about?

    Used to turn a free-text assertion into a testable claim on a real day,
    instead of inventing a concept out of a sentence fragment.
    """
    return [h.day for h in search_curriculum(curriculum, text, limit=limit) if h.score >= 3.0]


# ---------------------------------------------------------------------------
# LangChain tool bindings — the agentic loop
# ---------------------------------------------------------------------------


def build_tools(curriculum: Curriculum, profile: Profile) -> list[Any]:
    """Expose the four functions above as tools a model can call itself.

    The director does not need this: it already knows which day it selected and
    reads it directly, which is faster and cannot fail. This exists so the
    interviewer can *investigate* — follow a thread the candidate opened into a
    day nobody planned to visit.
    """
    from langchain_core.tools import tool

    @tool
    def curriculum_search(query: str) -> str:
        """Find which cohort days cover a topic, tool or technique."""
        hits = search_curriculum(curriculum, query)
        if not hits:
            return "No matching days."
        return "\n".join(f"day {h.day}: {h.title} [{h.type}] (matched {', '.join(h.matched)})" for h in hits)

    @tool
    def curriculum_read(day: int) -> str:
        """Read the full objectives and tooling for one cohort day."""
        d = read_day(curriculum, day)
        if "error" in d:
            return d["error"]
        objectives = "\n".join(f"- {o}" for o in d["objectives"])
        return f"Day {d['day']}: {d['title']} [{d['type']}]\nTools: {', '.join(d['tools'])}\n{objectives}"

    @tool
    def curriculum_list(module: int | None = None) -> str:
        """List cohort days, optionally filtered to one module (1-8)."""
        return "\n".join(f"day {d['day']}: {d['title']}" for d in list_days(curriculum, module))

    @tool
    def record_lookup(day: int) -> str:
        """Check what this candidate did on a given cohort day."""
        r = candidate_record(profile, day)
        return "; ".join(f"{k}={v}" for k, v in r.items() if v is not None)

    return [curriculum_search, curriculum_read, curriculum_list, record_lookup]
