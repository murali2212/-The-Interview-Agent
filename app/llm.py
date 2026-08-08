"""Unified model interface.

One decision lives here: live model or deterministic engine. Everything above
this file calls `ask` / `ask_json` and never learns which one answered.

Chains are built with LCEL (`prompt | model | parser`) so the pipeline is a
single composed object rather than glue code.

Nothing in this module raises. A failed call degrades to the offline engine,
because an interview that 500s mid-question is worse than one that asks a
slightly plainer question.
"""

from __future__ import annotations

import json
import logging
import re
import time
from functools import lru_cache
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from . import config

log = logging.getLogger("abtalks.llm")

TIER_MAIN = "main"
TIER_FAST = "fast"


@lru_cache(maxsize=4)
def _model(tier: str):
    """Cached so we build one client per tier, not one per turn."""
    if not config.llm_available():
        return None
    try:
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.GROQ_MODEL_MAIN if tier == TIER_MAIN else config.GROQ_MODEL_FAST,
            temperature=config.LLM_TEMPERATURE,
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=config.LLM_MAX_RETRIES,
        )
    except Exception as exc:  # missing package, bad key, anything
        log.warning("Groq unavailable, falling back to offline engine: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
#
# When the quota is gone, every call still pays the full timeout before falling
# back, and an eight-question interview turns into minutes of dead air. After a
# few consecutive failures we stop calling out at all for a cooldown, serve the
# deterministic engine, and try again later.

_consecutive_failures = 0
_open_until = 0.0


def _breaker_open() -> bool:
    return time.monotonic() < _open_until


def _record_failure() -> None:
    global _consecutive_failures, _open_until
    _consecutive_failures += 1
    if _consecutive_failures >= config.LLM_FAILURE_THRESHOLD:
        _open_until = time.monotonic() + config.LLM_COOLDOWN_SECONDS
        log.warning(
            "Live model dropped for %ss after %d consecutive failures; serving offline engine",
            config.LLM_COOLDOWN_SECONDS,
            _consecutive_failures,
        )
        _consecutive_failures = 0


def _record_success() -> None:
    global _consecutive_failures
    _consecutive_failures = 0


def is_live() -> bool:
    return _model(TIER_MAIN) is not None and not _breaker_open()


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def ask(
    prompt: ChatPromptTemplate,
    variables: dict[str, Any],
    *,
    tier: str = TIER_MAIN,
    fallback: str,
) -> str:
    """Run a prompt and return prose. Returns `fallback` on any failure."""
    model = _model(tier)
    if model is None or _breaker_open():
        return fallback
    try:
        chain = prompt | model | StrOutputParser()
        out = (chain.invoke(variables) or "").strip()
        _record_success()
        return out or fallback
    except Exception as exc:
        _record_failure()
        log.warning("LLM call failed (%s); using fallback", type(exc).__name__)
        return fallback


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_loose_json(raw: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """Models wrap JSON in prose and fences no matter how firmly you ask."""
    if not raw:
        return fallback

    candidates: list[str] = []
    fenced = _FENCE.search(raw)
    if fenced:
        candidates.append(fenced.group(1))
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])
    candidates.append(raw)

    for text in candidates:
        for attempt in (text, re.sub(r",(\s*[}\]])", r"\1", text)):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return fallback


def ask_json(
    prompt: ChatPromptTemplate,
    variables: dict[str, Any],
    *,
    tier: str = TIER_MAIN,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    raw = ask(prompt, variables, tier=tier, fallback="")
    if not raw:
        return fallback
    return parse_loose_json(raw, fallback)


# ---------------------------------------------------------------------------
# Speakability
# ---------------------------------------------------------------------------

_MARKDOWN = re.compile(r"[*_`#>]+")
_BULLET = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s*", re.M)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def speakable(text: str, max_sentences: int | None = None) -> str:
    """Strip anything a text-to-speech engine would read aloud as punctuation,
    and hold the reply to a length someone can actually listen to.

    Applied to the text path as well as the voice path, so the same string
    works in both and we never discover the difference at demo time.
    """
    if not text:
        return ""
    cleaned = _BULLET.sub("", _MARKDOWN.sub("", text)).replace("\n", " ")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip().strip('"')

    limit = max_sentences or config.MAX_REPLY_SENTENCES
    parts = [p.strip() for p in _SENTENCE.split(cleaned) if p.strip()]
    if len(parts) > limit:
        # Keep the question. It is almost always last.
        question = next((p for p in reversed(parts) if p.endswith("?")), parts[-1])
        kept = parts[: limit - 1] + [question]
        cleaned = " ".join(dict.fromkeys(kept))
    return cleaned.strip()


def describe() -> dict[str, Any]:
    live = is_live()
    return {
        "provider": "groq" if live else "deterministic-offline",
        "live": live,
        "main_model": config.GROQ_MODEL_MAIN if live else None,
        "fast_model": config.GROQ_MODEL_FAST if live else None,
    }
