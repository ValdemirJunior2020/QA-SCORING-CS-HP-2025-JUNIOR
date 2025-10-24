# backend/gemini_utils.py
import os
import google.generativeai as genai

# Read key from environment (preferred)
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Configure only if we have a key
if API_KEY:
    genai.configure(api_key=API_KEY)

MODEL_NAME = "models/gemini-2.5-flash"

def analyze_with_gemini(masked_transcript: str) -> str:
    if not masked_transcript:
        return "Transcript was empty."
    if not API_KEY:
        return ("Gemini API key missing. Set GEMINI_API_KEY environment variable "
                "and restart the backend.")

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
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        return text.strip() if text.strip() else "No AI suggestions available."
    except Exception as e:
        return f"AI analysis error: {e}"
