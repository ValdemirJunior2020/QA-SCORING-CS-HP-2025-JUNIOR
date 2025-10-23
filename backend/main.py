from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, json, whisper, os
from fuzzywuzzy import fuzz
from gemini_utils import analyze_with_gemini

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

with open("qa_criteria.json", "r") as f:
    qa_data = json.load(f)

model = whisper.load_model("base")

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    audio_path = f"temp_{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    result = model.transcribe(audio_path)
    transcript = result["text"]

    total_score = 0
    max_score = sum(c["score"] for c in qa_data["criteria"])
    details = []

    for c in qa_data["criteria"]:
        found = any(
            fuzz.partial_ratio(k.lower(), transcript.lower()) > 70
            for k in c["keywords"] + c["alternative_phrases"]
        )
        details.append({
            "id": c["id"],
            "description": c["description"],
            "passed": found,
            "score": c["score"] if found else 0
        })
        if found:
            total_score += c["score"]

    percentage = round((total_score / max_score) * 100, 2)
    passing = percentage >= 90
    ai_feedback = analyze_with_gemini(transcript)

    os.remove(audio_path)
    return {
        "transcript": transcript,
        "score": percentage,
        "passing": passing,
        "details": details,
        "ai_feedback": ai_feedback
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
