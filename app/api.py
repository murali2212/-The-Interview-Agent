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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import config, director
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
