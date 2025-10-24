from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper
import os
import json
import re
import asyncio
from gemini_utils import analyze_with_gemini

app = FastAPI()

# CORS (relax during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to ["http://localhost:3000"] later if you like
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load criteria ----------
base_path = os.path.dirname(__file__)
criteria_path = os.path.join(base_path, "qa_criteria.json")
if not os.path.exists(criteria_path):
    raise FileNotFoundError(f"QA criteria file not found at {criteria_path}")

with open(criteria_path, "r", encoding="utf-8") as f:
    qa_criteria = json.load(f)

# ---------- Load Whisper (fast) ----------
print("Loading Whisper model (tiny)...")
model = whisper.load_model("tiny")
print("Whisper model loaded.")

# ---------- PII Sanitization Utilities ----------

ITINERARY_RE = re.compile(r"\bH\d{6,12}\b")  # preserve these tokens

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    flags=re.IGNORECASE,
)

# Phones: +1 (optional), (###) ###-####, ###-###-####, ###.###.####, etc.
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}(?!\w)"
)

# Names after cues: “my name is John Doe”, “guest name is Jane Smith”, “this is Mary”
NAME_CUE_RE = re.compile(
    r"(?i)\b(my name is|guest name is|this is|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
)

# Generic long digit sequences (13–19 digits; may include separators)
LONG_DIGIT_RE = re.compile(
    r"(?<![A-Z])(?:\d[ -]?){13,19}(?!\w)"
)
# ^ Note (?<![A-Z]) prevents matching if preceded by 'H' (so H123456 stays)
# We'll still do a Luhn check to confirm it's a card before masking.


def luhn_check(number: str) -> bool:
    """Return True if number (digits only) passes Luhn checksum."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    parity = (len(digits) - 2) % 2
    for i, d in enumerate(digits[::-1]):
        if i % 2 == 0:
            checksum += d
        else:
            doubled = d * 2
            checksum += doubled - 9 if doubled > 9 else doubled
    return checksum % 10 == 0


def redact_emails(text: str) -> str:
    return EMAIL_RE.sub("[EMAIL]", text)


def normalize_spoken_email(text: str) -> str:
    """
    Try to catch 'spoken' email patterns like:
    'john dot doe at aol dot com' -> 'john.doe@aol.com' (then EMAIL_RE handles it)
    This is a best-effort heuristic.
    """
    # collapse patterns " at " -> "@", " dot " -> ".", remove spaces around @ and .
    t = re.sub(r"(?i)\s+at\s+", "@", text)
    t = re.sub(r"(?i)\s+dot\s+", ".", t)
    # also collapse spelled addresses like "a o l . com" -> "aol.com"
    t = re.sub(r"(?i)\b([a-z])\s+(?=[a-z]\b)", r"\1", t)  # join single letters
    t = re.sub(r"(?i)(@)\s+", r"\1", t)
    t = re.sub(r"(?i)\.\s+", ".", t)
    return t


def redact_phones(text: str) -> str:
    return PHONE_RE.sub("[PHONE]", text)


def redact_names_after_cues(text: str) -> str:
    def repl(m):
        cue = m.group(1)
        return f"{cue} [NAME]"
    return NAME_CUE_RE.sub(repl, text)


def redact_cards(text: str) -> str:
    def repl(m):
        s = m.group(0)
        # Skip if inside an itinerary (we already avoided with (?<![A-Z]), but double-check)
        if ITINERARY_RE.search(s):
            return s
        # Extract digits and check Luhn
        digits = "".join(ch for ch in s if ch.isdigit())
        if luhn_check(digits):
            return "[CARD]"
        return s  # not a valid card, leave it
    return LONG_DIGIT_RE.sub(repl, text)


def sanitize_transcript(raw: str) -> str:
    """
    Redact PII but keep itinerary numbers (H + digits).
    Order matters: normalize spoken emails -> redact emails -> redact cards -> phones -> names.
    """
    if not raw:
        return raw

    # Don’t log raw transcripts to console to avoid leaking PII.
    text = raw

    # Preserve itinerary tokens by temporarily protecting them
    itineraries = {}
    def protect_itin(m):
        token = m.group(0)
        key = f"__ITIN__{len(itineraries)}__"
        itineraries[key] = token
        return key

    text = ITINERARY_RE.sub(protect_itin, text)

    # Normalize and redact
    text = normalize_spoken_email(text)
    text = redact_emails(text)
    text = redact_cards(text)
    text = redact_phones(text)
    text = redact_names_after_cues(text)

    # Restore itinerary tokens
    for key, val in itineraries.items():
        text = text.replace(key, val)

    return text

# ---------- Scoring ----------
def compute_score(transcript: str, criteria: dict) -> float:
    total = 0
    earned = 0
    t = transcript.lower()

    for item in criteria.get("criteria", []):
        points = item.get("score", 0)
        total += points
        # simple keyword presence
        kws = [kw.lower() for kw in item.get("keywords", [])]
        if any(kw in t for kw in kws):
            earned += points

    return round((earned / total) * 100, 2) if total > 0 else 0.0


@app.get("/")
async def root():
    return {"message": "QA Scoring API is running!"}


@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    """
    1) Save audio
    2) Transcribe
    3) SANITIZE transcript (mask PII, keep H########)
    4) Score
    5) Gemini on masked transcript
    """
    audio_path = os.path.join(base_path, "temp_audio.mp3")

    try:
        # Save uploaded audio
        with open(audio_path, "wb") as buffer:
            buffer.write(await file.read())

        # Transcribe
        result = model.transcribe(audio_path)
        raw_transcript = result.get("text", "").strip()
        if not raw_transcript:
            return {
                "qa_score": 0,
                "transcript": "",
                "ai_suggestions": "No speech detected."
            }

        # SANITIZE before anything else (display + Gemini)
        masked_transcript = sanitize_transcript(raw_transcript)

        # Score against criteria (use the masked transcript to be safe)
        qa_score = compute_score(masked_transcript, qa_criteria)

        # Gemini analysis with masked transcript, with timeout safety
        try:
            ai_suggestions = await asyncio.wait_for(
                asyncio.to_thread(analyze_with_gemini, masked_transcript),
                timeout=40
            )
        except asyncio.TimeoutError:
            ai_suggestions = "AI analysis took too long. Please retry."

        return {
            "qa_score": qa_score,
            "transcript": masked_transcript,
            "ai_suggestions": ai_suggestions or "No AI suggestions available."
        }

    except Exception as e:
        print(f"Error in /upload/: {e}")
        return {
            "qa_score": 0,
            "transcript": "",
            "ai_suggestions": "Error analyzing audio."
        }

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
