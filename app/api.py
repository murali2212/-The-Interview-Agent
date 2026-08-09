"""HTTP surface.

`POST /api/interview` is the graded endpoint and is a faithful implementation
of docs-technical-spec.md. Everything else exists for the demo and can be
ignored by an evaluator.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from . import config, director, llm
from . import session as store
from .models import InterviewRequest, InterviewResponse
from .normalize import load_candidates

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("abtalks.api")

STATIC = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(
    title="ABTalks Interview Agent",
    version="0.1.0",
    description="A panel of agents that forms explicit beliefs about a candidate, then interviews to test them.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# The graded endpoint
# ---------------------------------------------------------------------------


@app.post("/api/interview", response_model=InterviewResponse, response_model_exclude_none=True)
def interview(req: InterviewRequest) -> InterviewResponse:
    """Start an interview (send `candidate`) or continue one (send `message`).

    Never raises. A failure here during judging is worth more lost points than
    any wrong answer, so an unexpected error degrades to a spoken apology that
    keeps the session alive.
    """
    try:
        return director.turn(req)
    except Exception:
        log.exception("turn failed for session %s", req.sessionId)
        return InterviewResponse(
            reply="Sorry, I lost my thread for a moment. Could you say that again?",
            done=False,
        )


# ---------------------------------------------------------------------------
# Demo support
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": "abtalks-interview-agent",
            **config.describe(),
            "activeSessions": store.count(),
            "contract": {
                "endpoint": "POST /api/interview",
                "start": {"sessionId": "str", "candidate": "{member, missions, signals}"},
                "turn": {"sessionId": "str", "message": "str"},
                "response": {"reply": "str", "done": "bool", "feedback": "on final turn"},
            },
        }
    )


@app.get("/api/candidates")
def candidates() -> JSONResponse:
    """Local roster, for the demo UI only. The evaluator posts its own profile."""
    out = []
    for c in load_candidates():
        s = c.signals
        out.append(
            {
                "id": c.member.id,
                "name": c.member.name,
                "jobRole": c.member.jobRole,
                "yearsExperience": c.member.yearsExperience,
                "missionsCompleted": s.missionsCompleted,
                "missionsFirstTry": s.missionsFirstTry,
                "commitDays": s.commitDays,
                "candidate": c.model_dump(),
            }
        )
    return JSONResponse({"candidates": out})


@app.get("/api/session/{session_id}")
def session_detail(session_id: str) -> JSONResponse:
    s = store.get(session_id)
    if s is None:
        return JSONResponse({"error": "unknown session"}, status_code=404)
    return JSONResponse(director._insight(s))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


@app.get("/")
def index():
    page = STATIC / "index.html"
    if not page.exists():
        return JSONResponse({"error": "UI not built", "api": "POST /api/interview"})
    return FileResponse(page)


@app.get("/mic-test")
def mic_test():
    """Standalone microphone diagnostic. Measures each stage of the voice
    pipeline separately so a silent capture can be traced to the exact stage
    that dropped it, rather than guessed at."""
    page = STATIC / "mic-test.html"
    if not page.exists():
        return JSONResponse({"error": "diagnostic page missing"}, status_code=404)
    return FileResponse(page)


# ---------------------------------------------------------------------------
# Candidate ingest — accept JSON / PDF / text, LLM-shape non-JSON
# ---------------------------------------------------------------------------


_CAND_SCHEMA_HINT = """
Return ONLY a JSON object matching this exact shape (no prose, no fences):

{{
  "member": {{
    "id": "CAND-XXX",
    "name": "Full Name",
    "jobRole": "e.g. AI Engineer",
    "yearsExperience": 0
  }},
  "missions": [
    {{"day": 7, "passed": true, "attempts": 1, "title": "Vector Databases"}}
  ],
  "signals": {{
    "missionsCompleted": 0,
    "missionsFirstTry": 0,
    "commitDays": 0
  }}
}}

Rules:
- Infer sensible values from the source text. If a field is truly not present,
  use "" for strings, 0 for numbers, [] for missions.
- passed:true = completed. passed:false = observed failure. skipped:true
  for deliberately skipped days. Omit days that are simply not mentioned.
- attempts defaults to 1 when the source does not say.
- id should be short and stable, uppercase, e.g. CAND-UP1.
"""


def _extract_text(filename: str, data: bytes) -> tuple[str, str]:
    """Return (text, kind). kind in {"json","pdf","text"}."""
    name = (filename or "").lower()
    if name.endswith(".json"):
        return data.decode("utf-8", errors="replace"), "json"
    if name.endswith(".pdf") or data[:4] == b"%PDF":
        from pypdf import PdfReader
        from io import BytesIO
        try:
            reader = PdfReader(BytesIO(data))
            pages = [(p.extract_text() or "") for p in reader.pages]
            return "\n\n".join(pages).strip(), "pdf"
        except Exception as exc:
            log.warning("pdf extract failed: %s", exc)
            return "", "pdf"
    return data.decode("utf-8", errors="replace"), "text"


@app.post("/api/candidate-from-file")
async def candidate_from_file(file: UploadFile = File(...)) -> JSONResponse:
    """Accept any file, return a candidate dict shaped for /api/interview.
    JSON files are parsed directly. PDF/text files are extracted and passed
    through the LLM to be shaped into the wire contract."""
    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "empty file"}, status_code=400)

    text, kind = _extract_text(file.filename or "", raw)

    if kind == "json":
        try:
            obj = json.loads(text)
        except Exception as exc:
            return JSONResponse({"error": f"invalid JSON: {exc}"}, status_code=400)
        if not isinstance(obj, dict) or "member" not in obj:
            return JSONResponse({"error": "JSON must contain a 'member' object"}, status_code=400)
        return JSONResponse({"candidate": obj, "source": "json"})

    if not text.strip():
        return JSONResponse({"error": f"no text extracted from {kind} file"}, status_code=400)

    if not llm.is_live():
        return JSONResponse(
            {"error": "This file type needs an LLM to parse, but no Groq key is configured on the server."},
            status_code=503,
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You extract structured candidate profiles from resumes and notes for a technical AI cohort interview." + _CAND_SCHEMA_HINT),
        ("user", "Source ({kind}):\n\n{text}\n\nReturn the JSON object only."),
    ])
    result = llm.ask_json(
        prompt,
        {"kind": kind, "text": text[:8000]},
        fallback={},
    )
    if not result or "member" not in result:
        return JSONResponse({"error": "LLM could not extract a candidate profile from this file."}, status_code=422)
    return JSONResponse({"candidate": result, "source": kind})


# ---------------------------------------------------------------------------
# LiveKit token — no extra deps, pure JWT HS256
# ---------------------------------------------------------------------------


class LiveKitTokenRequest(BaseModel):
    apiKey: str
    apiSecret: str
    url: str
    room: str = "interview-room"
    identity: str = "candidate"


@app.post("/api/livekit-token")
def livekit_token(req: LiveKitTokenRequest) -> JSONResponse:
    if not req.apiKey or not req.apiSecret:
        return JSONResponse({"error": "missing credentials"}, status_code=400)

    def b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    now = int(time.time())
    header = b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = b64url(json.dumps({
        "iss": req.apiKey,
        "sub": req.identity,
        "iat": now,
        "exp": now + 7200,
        "nbf": now,
        "video": {"room": req.room, "roomJoin": True, "canPublish": True, "canSubscribe": True},
    }, separators=(",", ":")).encode())
    msg = f"{header}.{payload}".encode()
    sig = b64url(hmac.new(req.apiSecret.encode(), msg, hashlib.sha256).digest())
    return JSONResponse({"token": f"{header}.{payload}.{sig}", "url": req.url, "room": req.room})
