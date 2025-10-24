from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper
import os
import json
import re
import asyncio
from rapidfuzz import fuzz

# Load .env before importing gemini_utils (so env is ready if you also use it at import)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from gemini_utils import analyze_with_gemini, gemini_health

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Paths / Criteria ---
BASE = os.path.dirname(__file__)
CRITERIA_PATH = os.path.join(BASE, "qa_criteria.json")
if not os.path.exists(CRITERIA_PATH):
    raise FileNotFoundError(f"QA criteria file not found: {CRITERIA_PATH}")

with open(CRITERIA_PATH, "r", encoding="utf-8") as f:
    QA_CRITERIA = json.load(f)

# --- Whisper (tiny for speed) ---
print("Loading Whisper model: tiny")
WHISPER_MODEL = whisper.load_model("tiny")
print("Whisper ready.")

# =========================
#      PII SANITIZATION
# =========================
ITINERARY_RE = re.compile(r"\bH\d{6,12}\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}(?!\w)")
NAME_CUE_RE = re.compile(r"(?i)\b(my name is|guest name is|this is|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})")
LONG_DIGIT_RE = re.compile(r"(?<![A-Z])(?:\d[ -]?){13,19}(?!\w)")

def luhn_check(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0

def normalize_spoken_email(text: str) -> str:
    t = re.sub(r"(?i)\s+at\s+", "@", text)
    t = re.sub(r"(?i)\s+dot\s+", ".", t)
    t = re.sub(r"(?i)\b([a-z])\s+(?=[a-z]\b)", r"\1", t)
    t = re.sub(r"(?i)(@)\s+", r"\1", t)
    t = re.sub(r"(?i)\.\s+", ".", t)
    return t

def redact_emails(text: str) -> str:
    return EMAIL_RE.sub("[EMAIL]", text)

def redact_phones(text: str) -> str:
    return PHONE_RE.sub("[PHONE]", text)

def redact_names_after_cues(text: str) -> str:
    def repl(m):
        return f"{m.group(1)} [NAME]"
    return NAME_CUE_RE.sub(repl, text)

def redact_cards(text: str) -> str:
    def repl(m):
        s = m.group(0)
        if ITINERARY_RE.search(s):  # keep itinerary IDs (e.g., H12345678)
            return s
        digits = "".join(ch for ch in s if ch.isdigit())
        if luhn_check(digits):
            return "[CARD]"
        return s
    return LONG_DIGIT_RE.sub(repl, text)

def sanitize_transcript(raw: str) -> str:
    if not raw:
        return raw
    t = raw
    placeholders = {}

    def protect_itin(m):
        key = f"__ITIN__{len(placeholders)}__"
        placeholders[key] = m.group(0)
        return key

    t = ITINERARY_RE.sub(protect_itin, t)

    t = normalize_spoken_email(t)
    t = redact_emails(t)
    t = redact_cards(t)
    t = redact_phones(t)
    t = redact_names_after_cues(t)

    for k, v in placeholders.items():
        t = t.replace(k, v)
    return t

# =========================
#   PARTICIPANT NAMES
# =========================
def mask_name(name: str) -> str:
    parts = name.split()
    masked = []
    for p in parts:
        if len(p) <= 1:
            masked.append(p[0] + "*")
        else:
            masked.append(p[0] + "*" * (len(p) - 1))
    return " ".join(masked)

def initials(name: str) -> str:
    return "".join([w[0].upper() + "." for w in name.split() if w])

def extract_participant_names(raw_text: str):
    if not raw_text:
        return {"agent": None, "customer": None}
    matches = list(NAME_CUE_RE.finditer(raw_text))
    if not matches:
        return {"agent": None, "customer": None}
    items = [(m.start(), m.group(1).lower(), m.group(2).strip()) for m in matches]

    agent_name = None
    customer_name = None

    for pos, cue, name in items:
        if agent_name is None and cue in ("my name is", "this is", "i am") and pos < 400:
            agent_name = name
            break
    for pos, cue, name in items:
        if customer_name is None and cue == "guest name is":
            customer_name = name
            break
    if customer_name is None:
        for _, _, name in items:
            if agent_name and name != agent_name:
                customer_name = name
                break
    if agent_name is None and items:
        agent_name = items[0][2]
    if customer_name is None and len(items) >= 2:
        customer_name = items[1][2]

    def pack(n):
        if not n:
            return None
        return {"masked": mask_name(n), "initials": initials(n)}

    return {"agent": pack(agent_name), "customer": pack(customer_name)}

# =========================
#      FUZZY SCORING
# =========================
def _is_multiword_or_long(s: str) -> bool:
    if not s:
        return False
    tokens = s.strip().split()
    return len(tokens) >= 2 or len(s.strip()) >= 6

def _best_fuzzy_match(phrases, haystack_lower: str, floor: int = 72):
    best_phrase, best_score, best_mode = None, 0, "none"
    for p in phrases:
        if not p:
            continue
        p_l = p.lower().strip()
        if p_l and p_l in haystack_lower:
            return (p, 100, "substring")
        if _is_multiword_or_long(p_l):
            s1 = fuzz.partial_ratio(p_l, haystack_lower)
            s2 = fuzz.token_set_ratio(p_l, haystack_lower)
            s = max(s1, s2)
            if s > best_score:
                best_phrase, best_score, best_mode = p, s, "fuzzy"
    if best_score >= floor:
        return (best_phrase, int(best_score), best_mode)
    return (None, 0, "none")

def score_with_breakdown(transcript: str, criteria: dict):
    t = (transcript or "").lower()
    total_points = 0
    earned_points = 0
    breakdown = []

    for item in criteria.get("criteria", []):
        cid = item.get("id")
        desc = item.get("description", "")
        guideline = item.get("guideline", "")
        points = int(item.get("score", 0))

        keywords = item.get("keywords", []) or []
        alt = item.get("alternative_phrases", []) or []
        exact_only = [k for k in keywords if not _is_multiword_or_long(k)]
        fuzzy_ok = [k for k in keywords if _is_multiword_or_long(k)]
        if guideline:
            fuzzy_ok.append(guideline)
        fuzzy_ok.extend(alt)

        total_points += points

        matched_phrase, similarity, mode, passed = None, 0, "none", False

        for k in exact_only:
            k_l = k.lower().strip()
            if k_l and k_l in t:
                matched_phrase, similarity, mode, passed = k, 100, "substring", True
                break

        if not passed and fuzzy_ok:
            mp, sc, md = _best_fuzzy_match(fuzzy_ok, t, floor=72)
            if mp and sc >= 72:
                matched_phrase, similarity, mode, passed = mp, sc, md, True

        if passed:
            earned_points += points

        breakdown.append({
            "id": cid,
            "description": desc,
            "guideline": guideline,
            "points": points,
            "passed": passed,
            "matched_phrase": matched_phrase,
            "similarity": similarity,
            "mode": mode
        })

    score_percent = round((earned_points / total_points) * 100, 2) if total_points > 0 else 0.0
    totals = {
        "earned_points": earned_points,
        "total_points": total_points,
        "passed": sum(1 for b in breakdown if b["passed"]),
        "failed": sum(1 for b in breakdown if not b["passed"]),
    }
    return score_percent, breakdown, totals

# =========================
#          ROUTES
# =========================
@app.get("/")
async def root():
    return {"message": "QA Scoring API is running"}

@app.get("/gemini/health")
async def gemini_health_check():
    return gemini_health()

@app.get("/debug/env")
async def debug_env():
    key = (os.getenv("GEMINI_API_KEY") or "")
    return {
        "key_present": bool(key),
        "key_prefix": (key[:6] + "…" if key else ""),
        "has_dotenv": True
    }

# Accept both /upload and /upload/
@app.post("/upload")
@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    allowed_exts = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
    orig_ext = os.path.splitext(file.filename or "")[1].lower()
    ext = orig_ext if orig_ext in allowed_exts else ".wav"
    audio_path = os.path.join(BASE, f"temp_audio{ext}")

    try:
        content = await file.read()
        with open(audio_path, "wb") as buffer:
            buffer.write(content)

        print(f"[upload] Transcribing {audio_path} ({len(content)} bytes)")
        result = WHISPER_MODEL.transcribe(audio_path)
        raw_transcript = (result.get("text") or "").strip()

        if not raw_transcript:
            return {
                "qa_score": 0,
                "transcript": "",
                "ai_suggestions": "No speech detected or audio could not be decoded.",
                "qa_breakdown": [],
                "qa_totals": {"earned_points": 0, "total_points": 0, "passed": 0, "failed": 0},
                "participants": {"agent": None, "customer": None}
            }

        participants = extract_participant_names(raw_transcript)
        masked_transcript = sanitize_transcript(raw_transcript)

        qa_score, qa_breakdown, qa_totals = score_with_breakdown(masked_transcript, QA_CRITERIA)

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
            "ai_suggestions": ai_suggestions or "No AI suggestions available.",
            "qa_breakdown": qa_breakdown,
            "qa_totals": qa_totals,
            "participants": participants
        }

    except Exception as e:
        print(f"[upload] ERROR: {e}")
        return {
            "qa_score": 0,
            "transcript": "",
            "ai_suggestions": f"Error analyzing audio: {e}",
            "qa_breakdown": [],
            "qa_totals": {"earned_points": 0, "total_points": 0, "passed": 0, "failed": 0},
            "participants": {"agent": None, "customer": None}
        }
    finally:
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
