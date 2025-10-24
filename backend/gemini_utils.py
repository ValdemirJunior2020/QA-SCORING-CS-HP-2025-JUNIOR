"""
Gemini helper: dynamic key loading, health checks, and analysis.

- Reads API key dynamically from env: GEMINI_API_KEY
- Adds /gemini/health diagnostics via gemini_health()
- Auto-selects a working model if the default isn't available
"""

import os
import google.generativeai as genai

PREFERRED_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-flash-latest",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
]

def _get_key() -> str:
    # Always read fresh from environment
    return (os.getenv("GEMINI_API_KEY") or "").strip()

def _configure():
    key = _get_key()
    if key:
        genai.configure(api_key=key)
    return key

def _pick_first_working_model() -> str:
    """
    Calls list_models and picks the first model that supports generateContent.
    Falls back to PREFERRED_MODELS order if list fails.
    """
    try:
        _configure()
        models = list(genai.list_models())
        names = {m.name: m for m in models}
        # Prefer our candidates if supported
        for name in PREFERRED_MODELS:
            m = names.get(name)
            if m and "generateContent" in getattr(m, "supported_generation_methods", []):
                return name
        # Else: first supported model
        for m in models:
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                return m.name
    except Exception:
        pass
    return PREFERRED_MODELS[0]

# Decide once which model name we’ll try first.
MODEL_NAME = _pick_first_working_model()

def analyze_with_gemini(masked_transcript: str) -> str:
    """
    Return concise AI coaching for the call.
    """
    if not masked_transcript:
        return "Transcript was empty."
    key = _get_key()
    if not key:
        return "Gemini API key missing. Set GEMINI_API_KEY and restart the backend."

    prompt = f"""
You are a QA coach for hotel reservations calls.

Transcript (PII-masked):
\"\"\"{masked_transcript}\"\"\"

Provide 4–6 bullet points:
- Call reason
- What the agent did well (map to behaviors)
- What was incorrect or missing (map to behaviors)
- Concrete, actionable improvements

Be concise and specific.
"""

    try:
        _configure()
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        return text.strip() if text.strip() else "No AI suggestions available."
    except Exception as e:
        return f"AI analysis error: {e}"

def gemini_health():
    """
    Returns a dict with key diagnostics and a minimal model call.
    Never exposes full key; only prefix for sanity checks.
    """
    key = _get_key()
    out = {
        "key_present": bool(key),
        "key_prefix": (key[:6] + "…" if key else ""),
        "selected_model": MODEL_NAME,
    }
    if not key:
        out["ok"] = False
        out["error"] = "No GEMINI_API_KEY visible to backend process."
        return out
    try:
        _configure()
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content("ping")
        out["ok"] = True
        out["response_preview"] = (getattr(resp, "text", "") or "")[:60]
        return out
    except Exception as e:
        out["ok"] = False
        out["error"] = str(e)
        return out
