"""Settings. Everything environment-dependent lands here and nowhere else."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

load_dotenv(ROOT / ".env")


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Language model
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Two tiers, one provider. The big model writes and judges; the small one does
# triage and summarisation where latency matters more than nuance.
GROQ_MODEL_MAIN = os.getenv("GROQ_MODEL_MAIN", "llama-3.3-70b-versatile")
GROQ_MODEL_FAST = os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))

#: Deliberately zero.
#:
#: Groq's free tier answers a 429 with a Retry-After that can be a full minute,
#: and the SDK honours it by sleeping inside the call. That turns one rate limit
#: into a sixty-second hang in the middle of an interview. We have an instant,
#: decent deterministic fallback, so failing fast and degrading beats waiting.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "0"))

#: After this many consecutive failures the live model is dropped for
#: COOLDOWN seconds. Without it every single call keeps paying the timeout while
#: the quota is exhausted, and the whole interview crawls.
LLM_FAILURE_THRESHOLD = int(os.getenv("LLM_FAILURE_THRESHOLD", "3"))
LLM_COOLDOWN_SECONDS = float(os.getenv("LLM_COOLDOWN_SECONDS", "45"))

#: When true, never call the network. The deterministic engine answers instead.
#: This is also the automatic behaviour when GROQ_API_KEY is missing.
OFFLINE_ONLY = _flag("OFFLINE_ONLY", default=False)


def llm_available() -> bool:
    return bool(GROQ_API_KEY) and not OFFLINE_ONLY


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# Sentence Transformers runs locally: no key, no quota, works on a plane.
# It is also what the cohort itself teaches on day 7, which keeps the project
# honest about the curriculum it is interviewing people on.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ENABLE_RETRIEVAL = _flag("ENABLE_RETRIEVAL", default=True)
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

# ---------------------------------------------------------------------------
# Interview shape
# ---------------------------------------------------------------------------

#: The interview is exactly this long. The brief asks for a minimum of eight
#: questions; running to twelve buys very little extra signal and costs the
#: candidate real time, so eight is both the floor and the ceiling.
#:
#: This makes the budget tight on purpose: eight questions must still reach
#: MIN_DISTINCT_DAYS across MIN_DISTINCT_MODULES, which means at most three
#: turns may revisit a day already covered. The director enforces that.
MIN_QUESTIONS = int(os.getenv("MIN_QUESTIONS", "8"))
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "8"))
MIN_DISTINCT_DAYS = int(os.getenv("MIN_DISTINCT_DAYS", "5"))

#: Distinct MODULES the interview must span.
#:
#: Day-count alone is not real coverage. A DevOps engineer whose role implies
#: days 27-31 will happily be interviewed on days 27, 28, 29, 30 and 31 — five
#: distinct days, one corner of the curriculum, and not a single question about
#: the embeddings, RAG or agent work they also completed.
MIN_DISTINCT_MODULES = int(os.getenv("MIN_DISTINCT_MODULES", "3"))

#: Hard floors from the brief, kept separate so tests assert against the rubric
#: rather than against whatever we happen to have configured.
RUBRIC_FLOOR_QUESTIONS = 8
RUBRIC_FLOOR_DAYS = 4

#: Spoken replies must stay short. This is a voice-first constraint applied to
#: the text path too, so the same string works in both transports.
MAX_REPLY_SENTENCES = int(os.getenv("MAX_REPLY_SENTENCES", "4"))

SEED = int(os.getenv("SEED", "1337"))

# ---------------------------------------------------------------------------
# Voice (phase two — read here so nothing needs restructuring later)
# ---------------------------------------------------------------------------

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "").strip()
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "").strip()
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "").strip()
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "").strip()


def voice_available() -> bool:
    return all(
        [DEEPGRAM_API_KEY, CARTESIA_API_KEY, LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]
    )


def describe() -> dict[str, object]:
    """Shown at /api/health so nobody has to guess how the process is wired."""
    return {
        "llm": {
            "provider": "groq" if llm_available() else "deterministic-offline",
            "live": llm_available(),
            "main_model": GROQ_MODEL_MAIN if llm_available() else None,
            "fast_model": GROQ_MODEL_FAST if llm_available() else None,
        },
        "retrieval": {"enabled": ENABLE_RETRIEVAL, "embedding_model": EMBEDDING_MODEL},
        "interview": {
            "min_questions": MIN_QUESTIONS,
            "max_questions": MAX_QUESTIONS,
            "min_distinct_days": MIN_DISTINCT_DAYS,
        },
        "voice_configured": voice_available(),
    }
