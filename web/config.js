// Backend base URL for this static frontend to call. Points at the deployed
// Railway backend by default so the Vercel-hosted copy of this file works
// with zero manual setup.
//
// Working against a local backend instead (e.g. testing app/ changes before
// pushing)? Don't edit this file — run in the browser console:
//   localStorage.setItem('apiBase', 'http://localhost:8000')
// It's read before window.API_BASE (see index.html) and persists across
// reloads; localStorage.removeItem('apiBase') to go back to this default.
window.API_BASE = "https://the-interview-agent-production-56db.up.railway.app";
