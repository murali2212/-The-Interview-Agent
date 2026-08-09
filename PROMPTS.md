# AI Usage Log

This project was built with [Claude Code](https://claude.com/claude-code) driving the actual
implementation — every file in this repo was written by Claude in response to a real prompt,
not scaffolded by hand and cleaned up after. This log exists so that's checkable rather than
just asserted.

Two views below: the **commit-by-commit build order** (pulled straight from `git log`, so it's
exactly what happened, not a summary written after the fact), and a **prompt-level log** of the
back-and-forth for the parts of the build that involved real iteration — debugging, design
decisions, and features that took more than one pass to get right.

---

## Commit-by-commit build order

```
360b77b  chore: project scaffold and technical specification
5c75c2b  feat(data): wire contract, curriculum and candidate normalisation
27892b3  feat(ledger): Bayesian evidence ledger and interview policy
7ce1f51  feat(llm): Groq via LCEL, prompt templates, deterministic fallback
ab32abd  feat(engine): profile analyst, panel director and agentic retrieval
eb34c68  feat(api): single POST /api/interview endpoint and web console
6515288  test: requirement audit, integration tests and documentation
ed2a537  feat: multi-page UI, voice mode with Whisper STT and best-voice TTS
2799dd2  chore: remove README, drop Deepgram/Cartesia from config and env example
5f3fdfc  docs: restore README with Mermaid architecture diagram
c3ec22e  feat: live captions + confirm-before-send in voice mode
e97e5d7  fix: voice mode wasn't hearing speech — dual mic grab was starving audio
dc79d73  fix: mic still silent — forced 16kHz sample rate was the real cause
4a0be3b  fix: remove Web Audio recording path; add mic diagnostic at /mic-test
1b72fad  Add candidate file upload: JSON, PDF, or text files
055cded  Add microphone device picker to voice interview setup
a30179f  Add demo auto-answer mode for recording a walkthrough video
e646447  Turn off demo auto-answer mode after recording
31d92b0  Remove demo auto-answer feature entirely
e737bd9  Add Vercel serverless deployment support
e91f68e  Split deployment: Supabase for session persistence, Vercel for static frontend
6726011  Add Procfile for Railway deployment
8ba4a75  Point web/config.js at the deployed Railway backend
```

---

## Prompt-level log

### Core engine (profile analyst, evidence ledger, director, offline fallback)
Built first, in one continuous pass: read the candidate/curriculum JSON contracts, implement a
Bayesian belief ledger keyed by curriculum day, a director that picks the next question by
expected information gain, an LLM layer over Groq with a deterministic offline fallback so the
interview can never die mid-judging, and the single `POST /api/interview` endpoint the whole
thing hangs off. `app/ledger.py`, `app/profile.py`, `app/director.py`, `app/llm.py`,
`app/offline.py`, `app/api.py`.

### Multi-page UI + voice mode
"Add voice mode — Groq Whisper for speech-to-text, browser SpeechSynthesis for text-to-speech."
Built the Welcome → Mode Selection → Candidate Picker → Interview page flow and the first voice
pipeline. `web/index.html`.

### "put inside the hamburger menu"
Coverage map / Evidence Ledger / Agent Trace sidebar was always visible during chat mode;
moved it behind a ☰ toggle so voice mode isn't cluttered with panels meant for reading, not
listening.

### "make the ai voice as real as possible... make it hear even if i speak in a low voice"
Two asks in one: best-available neural voice selection for TTS (ranked rule list preferring
Google/Microsoft neural voices over the browser default), and stronger mic sensitivity for STT.

### "i want to see the transcript of what i'm saying while i talk"
Live captions: the same recording is transcribed in rolling ~1.8s chunks via Groq Whisper so the
on-screen text is provably what the real pipeline captured, not a second independent guess.

### "it can't hear anything" — three-round mic debugging
The actual bug hunt, in the order it happened:
1. First hypothesis: a second, independent `SpeechRecognition` mic grab running alongside the
   real `getUserMedia` capture was starving the real one on Windows. Removed it — real issue,
   not the root cause.
2. Second hypothesis: a forced `sampleRate: 16000` constraint was causing near-silent capture on
   Windows Chrome. Removed the forced rate — also real, still not the root cause.
3. Root cause, found by tracing the actual signal path: recording was routed through a Web Audio
   `GainNode` → `MediaStreamAudioDestinationNode` graph before hitting `MediaRecorder`. That graph
   silently carries almost no signal under some Windows/Chrome configurations. Fixed by recording
   the raw `getUserMedia` stream directly — Web Audio is now used only for the VU meter and
   silence detection, never for what gets sent to Whisper. Also switched amplitude measurement
   from averaged FFT bins (buries speech under empty high-frequency bins) to true peak amplitude.
4. Built `/mic-test`, a standalone diagnostic page that measures every stage of the pipeline
   independently (device list, track state, raw vs. boosted amplitude, recorded byte count,
   playback, Whisper response) with a printed verdict — replacing further guessing with
   measurement.

### "check my microphone... peak level 0.0000 even though Windows settings are correct"
Ran the actual DevTools console check together — `getUserMedia` + `enumerateDevices()` — which
showed Chrome had silently picked a paired Bluetooth headset ("Communications - Headset") over
the wired USB mic Windows reported as default. Added a microphone device picker to the voice
setup modal so the candidate can force the right device instead of fighting Chrome's per-site
default.

### "add an option to add their own candidate data" → "make it file upload, not paste" → "let them upload anything, not just JSON"
Iterated three times to the final shape: a modal with a native file picker accepting `.json`,
`.pdf`, `.txt`, `.md`. JSON is used as-is; PDF/text is extracted and shaped into the interview
contract by the LLM against a strict schema prompt. New `POST /api/candidate-from-file` endpoint.

### "fill the answer automatically after 8 questions... when I say stop, stop"
A temporary demo-recording mode: the question is still spoken aloud and the mic still visibly
activates, but the transcription step is replaced with an LLM-generated plausible answer grounded
in the candidate's real profile, sent automatically so a full 8-question interview can run
unattended for a walkthrough video. Turned off, then fully deleted (endpoint and frontend path
both removed) once the recording was done — nothing demo-related remains in the codebase.

### "deploy this in vercel"
Before deploying: flagged that session state lived in a plain in-process Python dict, which is
fatal on Vercel's stateless serverless functions (a multi-turn interview could lose its session
between questions). Asked how to handle it; chose to add persistent storage rather than accept
the risk. First attempts deployed the whole FastAPI app to Vercel directly via file upload, which
kept hitting a payload size ceiling partway through the file set — abandoned in favor of a GitHub
import.

### "is there any place other than this to deploy"
Compared Railway, Render, Fly.io, and Cloud Run against Vercel's serverless constraints for this
specific app.

### "can we use supabase for backend and vercel for frontend"
Clarified scope first — Supabase can't run the ~120KB of existing Python interview logic
directly, so the split became: Python FastAPI backend on a persistent host (Railway), Supabase
Postgres as the session store (replacing the earlier Vercel-KV design), Vercel serving only the
static frontend. Rewired `app/session.py` to a PostgREST-backed store, added a configurable
`API_BASE` to the frontend (`web/config.js`), removed the now-obsolete Vercel Python entrypoint.

### Live deployment, done together in real time
Connected Supabase via MCP, created the `sessions` table, verified read/write/delete against the
real table. Deployed the backend to Railway (added a `Procfile` since Nixpacks needs an explicit
start command), verified `/api/health` and a full interview round-trip persisted correctly to
Supabase. Imported the frontend to Vercel — first attempt auto-detected the whole repo as a
FastAPI app and failed to build; corrected the Root Directory to `web` and the framework preset
to static, then verified the deployed frontend calling the deployed backend cross-origin, session
row visible in Supabase.

### "add face movement detection... 2 warnings then kick out with 'Suspicious movement detected multiple times'"
Given an exact escalation spec and Integrity Monitor panel layout. Built using MediaPipe's
official browser package (`@mediapipe/tasks-vision` via CDN — not a clone of the source repo,
which is a multi-gigabyte C++/mobile monorepo with no browser build of its own). Face-landmark
based head-turn detection, warning toast + live panel, kick-out screen. Found and fixed a real
bug during testing: the panel wasn't re-rendering after a warning event (only on kick-out),
caught by simulating the 1→2→3 escalation sequence directly rather than trusting the code by
inspection.

### "add PROMPTS.md"
This file, for the hackathon's AI-usage-log submission field.
