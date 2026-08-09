// Backend base URL for this static frontend to call. Empty string means
// same-origin (used when the backend still serves this file itself, e.g.
// local `uvicorn` dev). When the frontend is deployed separately (Vercel)
// from the backend (Railway/Render/Fly), edit the line below to the
// deployed backend's URL — no rebuild needed, this file is loaded as plain
// static JS.
window.API_BASE = "";
