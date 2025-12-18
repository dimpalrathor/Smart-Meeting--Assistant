# combined_backend.py - Backend for meeting planning & summarization
import os
import json
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pydub import AudioSegment
from faster_whisper import WhisperModel

# -----------------------------
# 1. GEMINI CONFIG
# -----------------------------
from google import genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Gemini API key missing. Set GEMINI_API_KEY env var.")

GEMINI_MODEL_NAME = "models/gemini-2.5-flash-lite"
GEMINI_MAX_TOKENS = 1200
GEMINI_TEMPERATURE = 0.0

# Initialize client
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"[Gemini] Initialized client for model: {GEMINI_MODEL_NAME}")
except Exception as e:
    raise RuntimeError(f"[Gemini] Client initialization failed: {e}")

# -----------------------------
# 2. FASTAPI APP
# -----------------------------
app = FastAPI(title="Smart Meeting Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import logging
logging.basicConfig(level=logging.INFO)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse({"error": "Internal Server Error"}, status_code=500)

# -----------------------------
# 3. PYDANTIC MODELS
# -----------------------------
class MeetingPlan(BaseModel):
    company_name: str
    title: str
    objective: str
    attendees: str
    duration: int
    focus_areas: str

class PreparationRequest(BaseModel):
    meeting_plan: MeetingPlan
    include_research: bool = True

# -----------------------------
# 4. WHISPER LOAD
# -----------------------------
print("Loading Whisper (tiny)...")
whisper_model = WhisperModel(
    "tiny", device="cpu", compute_type="int8",
    cpu_threads=os.cpu_count() or 4
)
print("Whisper loaded.")

# -----------------------------
# 5. AUDIO HELPERS
# -----------------------------
def convert_to_wav(path: Path) -> Path:
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    out = path.with_suffix(".wav")
    audio.export(out, format="wav")
    return out

def transcribe_audio(wav_path: Path):
    segments, _ = whisper_model.transcribe(
        str(wav_path),
        beam_size=1,
        vad_filter=True,
        vad_parameters={"threshold": 0.4},
    )
    transcript_parts, diarization = [], []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        transcript_parts.append(text)
        diarization.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": text,
        })
    return " ".join(transcript_parts).strip(), diarization

# -----------------------------
# 6. LOCAL FALLBACK
# -----------------------------
TASK_PATTERNS = [
    r"([A-Z][a-z]+)\s+(?:will|should|must|needs to)\s+(.+?)(?:\.|$)",
]
DEADLINE_PATTERNS = [
    r"by\s+[A-Za-z]+\s*\d*", r"by\s+end of week", r"by\s+tomorrow"
]

def local_extract(text: str) -> Dict[str, Any]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = " ".join(sentences[:3])
    action_points = []
    tasks, deadlines = [], []
    for s in sentences:
        sl = s.lower()
        if any(k in sl for k in ["should", "must", "will"]):
            action_points.append(s)
        for pat in TASK_PATTERNS:
            m = re.search(pat, s)
            if m:
                assignee, task = m.group(1), m.group(2)
                dl = None
                for dp in DEADLINE_PATTERNS:
                    md = re.search(dp, s, re.IGNORECASE)
                    if md:
                        dl = md.group(0)
                        deadlines.append(dl)
                        break
                tasks.append({
                    "speaker": assignee,
                    "assignee": assignee,
                    "task": task,
                    "deadline": dl,
                    "source": s
                })
    return {
        "summary": summary or "No summary available.",
        "action_points": list(dict.fromkeys(action_points)),
        "tasks": tasks,
        "deadlines": list(dict.fromkeys(deadlines)),
        "speakers": []
    }

# -----------------------------
# 7. GEMINI CALLS
# -----------------------------
def gemini_extract_structured(text: str) -> Dict[str, Any]:
    prompt = f"""
You are an expert meeting assistant. Return ONLY valid JSON with these keys:
summary, action_points, tasks, deadlines, speakers.
Transcript:
{text}
"""
    resp = client.generate_text(
        model=GEMINI_MODEL_NAME,
        input=prompt,
        max_output_tokens=GEMINI_MAX_TOKENS,
        temperature=0.1,
    )
    content = resp.text or ""
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Gemini JSON not returned.")
    parsed = json.loads(content[start:end+1])
    parsed.setdefault("summary", "")
    parsed.setdefault("action_points", [])
    parsed.setdefault("tasks", [])
    parsed.setdefault("deadlines", [])
    parsed.setdefault("speakers", [])
    return parsed

def gemini_generate_followup_email(summary: str, tasks: List[Dict[str, Any]], meeting_title: str) -> str:
    prompt = f"""
Meeting Title: {meeting_title}
Summary:
{summary}
Tasks JSON:
{json.dumps(tasks, indent=2)}
"""
    resp = client.generate_text(
        model=GEMINI_MODEL_NAME,
        input=prompt,
        max_output_tokens=GEMINI_MAX_TOKENS,
        temperature=GEMINI_TEMPERATURE,
    )
    return (resp.text or "").strip()

def gemini_generate_whatsapp(summary: str) -> str:
    prompt = f"Create a short WhatsApp style recap:\n{summary}"
    resp = client.generate_text(
        model=GEMINI_MODEL_NAME,
        input=prompt,
        max_output_tokens=300,
        temperature=0.2,
    )
    return (resp.text or "").strip()

def gemini_translate_summary(summary: str, target_lang: str) -> str:
    prompt = f"Translate the following summary into {target_lang}:\n{summary}"
    resp = client.generate_text(
        model=GEMINI_MODEL_NAME,
        input=prompt,
        max_output_tokens=400,
        temperature=0.0,
    )
    return (resp.text or "").strip()

# -----------------------------
# 8. SUMMARY ENDPOINT
# -----------------------------
@app.post("/summarize")
async def summarize_meeting(
    audio: UploadFile = File(...),
    meeting_title: Optional[str] = Form(None),
    target_lang: Optional[str] = Form(None),
):
    tmp_file = Path(tempfile.gettempdir()) / f"up_{audio.filename}"
    with open(tmp_file, "wb") as f:
        while chunk := await audio.read(1024 * 1024):
            f.write(chunk)

    wav = convert_to_wav(tmp_file)
    transcript, diarization = transcribe_audio(wav)

    if not transcript:
        raise HTTPException(status_code=400, detail="Empty transcription")

    try:
        extracted = gemini_extract_structured(transcript)
    except Exception:
        extracted = local_extract(transcript)

    summary_text = extracted.get("summary", "")

    followup_email = gemini_generate_followup_email(summary_text, extracted.get("tasks", []), meeting_title or "")
    whatsapp_msg = gemini_generate_whatsapp(summary_text)

    translated = None
    if target_lang:
        translated = gemini_translate_summary(summary_text, target_lang)

    # Cleanup
    tmp_file.unlink(missing_ok=True)
    wav.unlink(missing_ok=True)

    return {
        "status": "success",
        "transcript": transcript,
        "diarization": diarization,
        "summary": summary_text,
        "action_points": extracted.get("action_points", []),
        "tasks": extracted.get("tasks", []),
        "deadlines": extracted.get("deadlines", []),
        "speakers": extracted.get("speakers", []),
        "structured_summary": summary_text,
        "followup_email": followup_email,
        "whatsapp": whatsapp_msg,
        "translated_summary": translated,
        "target_lang": target_lang,
    }

# -----------------------------
# 9. RUN SERVER
# -----------------------------
@app.get("/")
async def root():
    return {"status": "online", "service": "Smart Meeting Assistant API"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("combined_backend:app", host="0.0.0.0", port=port, log_level="info")
