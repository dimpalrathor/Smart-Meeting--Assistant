# combined_backend.py - Complete backend for meeting planning, preparation & summarization
# Whisper-tiny + Gemini 2.0 Flash with structured meeting analysis

import os
import json
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pydub import AudioSegment
from faster_whisper import WhisperModel

# -----------------------------
# 1. GEMINI CONFIG
# -----------------------------
from google import genai


# Hardcoded Gemini API key (replace with your own if needed)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Gemini API key is missing. Set GEMINI_API_KEY env var.")

GEMINI_MODEL_NAME = "models/gemini-2.5-flash-lite"
GEMINI_MAX_TOKENS = 1200
GEMINI_TEMPERATURE = 0.0

# Upload limit (2GB)
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024

# Validate key
if not GEMINI_API_KEY:
    raise RuntimeError(
        "Gemini API key missing. Set environment variable GEMINI_API_KEY "
        "or set GEMINI_API_KEY in this file."
    )

# Gemini setup
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f" Gemini initialized with model: {GEMINI_MODEL_NAME}")
except Exception as e:
    raise RuntimeError(f"Failed to configure Gemini: {e}")


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
async def all_exception_handler(request, exc):
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
# 4. WHISPER (TINY) LOAD
# -----------------------------
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"

print(f"Loading faster-whisper (tiny) | device={WHISPER_DEVICE} compute={WHISPER_COMPUTE} ...")
whisper_model = WhisperModel(
    "tiny",
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE,
    cpu_threads=os.cpu_count() or 4,
)
print(" Whisper (tiny) loaded.")


# -----------------------------
# 5. AUDIO HELPERS
# -----------------------------
def convert_to_wav(path: Path) -> Path:
    """Convert any supported audio file to mono 16k WAV."""
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    out = path.with_suffix(".wav")
    audio.export(out, format="wav")
    return out


def transcribe_audio(wav_path: Path):
    """
    Transcribe audio using faster-whisper tiny and return diarization segments.
    Each segment: {start, end, text}
    """
    segments, info = whisper_model.transcribe(
        str(wav_path),
        beam_size=1,
        vad_filter=True,
        vad_parameters={"threshold": 0.4},
    )

    transcript_parts = []
    diarization: List[Dict[str, Any]] = []

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        transcript_parts.append(text)
        diarization.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
            }
        )

    transcript = " ".join(transcript_parts)
    return transcript.strip(), diarization


# -----------------------------
# 6. LOCAL FALLBACK EXTRACTOR
# -----------------------------
TASK_PATTERNS = [
    r"([A-Z][a-z]+)\s+(?:will|should|needs to|is going to|must|has to)\s+(.+?)(?:\.|$)",
    r"([A-Z][a-z]+)\s*:\s*(?:will|should|needs to|must)\s+(.+?)(?:\.|$)",
    r"([A-Z][a-z]+)\s+(?:assigned|tasked)\s+to\s+(.+?)(?:\.|$)",
]

DEADLINE_PATTERNS = [
    r"by\s+[A-Za-z]+\s*\d*",
    r"by\s+end of week",
    r"by\s+tomorrow",
    r"by\s+next week",
    r"by\s+Friday",
    r"by\s+Monday",
    r"by\s+Thursday",
    r"by\s+Sunday",
    r"before\s+next\s+meeting",
]


def local_extract(text: str) -> Dict[str, Any]:
    """Fallback extraction when Gemini fails."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = " ".join(sentences[:3])

    action_points: List[str] = []
    tasks: List[Dict[str, Optional[str]]] = []
    deadlines: List[str] = []

    for s in sentences:
        sl = s.lower()

        # Action point detector
        if any(k in sl for k in ["should", "need to", "must", "will", "targeting", "aim to"]):
            action_points.append(s)

        # Task patterns
        for pat in TASK_PATTERNS:
            m = re.search(pat, s)
            if m:
                assignee = m.group(1)
                task = m.group(2)
                deadline = None
                for dp in DEADLINE_PATTERNS:
                    md = re.search(dp, s, re.IGNORECASE)
                    if md:
                        deadline = md.group(0)
                        deadlines.append(deadline)
                        break
                tasks.append(
                    {
                        "speaker": assignee,
                        "assignee": assignee,
                        "task": task.strip(),
                        "deadline": deadline,
                        "source": s,
                    }
                )

        # Deadlines even if no tasks
        for dp in DEADLINE_PATTERNS:
            md = re.search(dp, s, re.IGNORECASE)
            if md:
                deadlines.append(md.group(0))

    return {
        "summary": summary.strip() or "No summary available.",
        "action_points": list(dict.fromkeys(action_points)),
        "tasks": tasks,
        "deadlines": list(dict.fromkeys(deadlines)),
        "speakers": [],
    }


# -----------------------------
# 7. GEMINI STRUCTURED EXTRACT
# -----------------------------
def gemini_extract_structured(text: str) -> Dict[str, Any]:
    """
    Use Gemini to extract structured summary from meeting transcript.
    Returns JSON with summary, action_points, tasks, deadlines, speakers.
    """
    if GEMINI_MODEL is None:
        raise RuntimeError("Gemini model not initialized.")

    prompt = f"""
You are an expert AI meeting assistant.

Read the following meeting transcript and return ONLY valid JSON with EXACTLY these keys:

{{
  "summary": "string, 3–6 sentences summarizing the meeting",
  "action_points": [
    "bullet point action item 1",
    "bullet point action item 2"
  ],
  "tasks": [
    {{
      "speaker": "who spoke this task (e.g., John, Priya, Lead)",
      "assignee": "who is responsible (can be same as speaker)",
      "task": "what must be done",
      "deadline": "short deadline phrase like 'Thursday', 'end of this week', or null if no deadline",
      "source": "short quote from the transcript that supports this task"
    }}
  ],
  "deadlines": [
    "Thursday",
    "Monday",
    "end of this week"
  ],
  "speakers": [
    "unique speaker name 1",
    "unique speaker name 2"
  ]
}}

Rules:
- Use names exactly as they appear in the transcript (John, Priya, Amit, Sarah, Lead, etc.).
- If someone introduces themselves (e.g., "Hi, I'm Rahul"), detect that and add "Rahul" as a speaker.
- Do NOT hallucinate tasks or deadlines that are not implied by the transcript.
- If a task has no clear deadline, set "deadline": null.
- "speakers" must be a de-duplicated list of canonical speaker names detected from the transcript.
- Return ONLY JSON. No markdown, no extra commentary.

Transcript:
{text}
"""

    try:
        response = GEMINI_MODEL.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": GEMINI_MAX_TOKENS,
            },
        )
        content = response.text or ""

        # Try to isolate JSON object
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Gemini did not return a JSON object.")

        json_str = content[start: end + 1]
        parsed = json.loads(json_str)

        # Sanity defaults
        parsed.setdefault("summary", "No summary available.")
        parsed.setdefault("action_points", [])
        parsed.setdefault("tasks", [])
        parsed.setdefault("deadlines", [])
        parsed.setdefault("speakers", [])

        return parsed

    except Exception as e:
        raise RuntimeError(f"Gemini call failed: {e}")


# -----------------------------
# 8. GEMINI EXTRA FEATURES
# -----------------------------
def gemini_generate_followup_email(summary: str, tasks: List[Dict[str, Any]], meeting_title: str = "") -> str:
    """Generate a professional follow-up email for the meeting."""
    if GEMINI_MODEL is None:
        return "Gemini not available for email generation."

    prompt = f"""
You are an expert meeting assistant.

Write a professional follow-up email based on the meeting summary and tasks below.

Meeting Title: {meeting_title or "Team Meeting"}

Rules:
- Tone: clear, concise, corporate, but friendly.
- Structure:
  - Subject line suggestion.
  - Greeting.
  - 2–3 lines summarizing meeting purpose and outcomes.
  - Bullet list of action items with owners & deadlines.
  - Closing line thanking participants and inviting questions.
- Do NOT hallucinate information that is not in the summary or tasks.
- Use only the details provided.

MEETING SUMMARY:
{summary}

TASKS (JSON):
{json.dumps(tasks, indent=2)}

Return only the email body text (no markdown, no JSON).
"""

    try:
        resp = GEMINI_MODEL.generate_content(
            prompt,
            generation_config={
                "temperature": GEMINI_TEMPERATURE,
                "max_output_tokens": GEMINI_MAX_TOKENS,
            },
        )
        return (resp.text or "").strip()
    except Exception as e:
        return f"Email generation failed: {e}"


def gemini_generate_whatsapp(summary: str) -> str:
    """Generate a short WhatsApp-style recap message."""
    if GEMINI_MODEL is None:
        return "Gemini not available for WhatsApp summary."

    prompt = f"""
Convert the following meeting summary into a short WhatsApp-style recap message.

Rules:
- 3–6 short bullet-like lines.
- Simple language.
- Add relevant emojis (2–6 total) where natural.
- Focus on decisions and key next steps.
- No greeting or signature.
- No markdown bullets, just plain text lines.

MEETING SUMMARY:
{summary}

Return only the message text.
"""
    try:
        resp = GEMINI_MODEL.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 300,
            },
        )
        return (resp.text or "").strip()
    except Exception as e:
        return f"WhatsApp summary failed: {e}"


def gemini_translate_summary(summary: str, target_lang: str) -> str:
    """Translate summary into target_lang when requested."""
    if GEMINI_MODEL is None:
        return summary

    if not target_lang:
        return summary

    prompt = f"""
Translate the meeting summary below into {target_lang}.

Rules:
- Preserve meaning exactly.
- Keep it concise and natural for native speakers.
- No extra explanations or comments.

SUMMARY:
{summary}
"""

    try:
        resp = GEMINI_MODEL.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 400,
            },
        )
        text = (resp.text or "").strip()
        return text or summary
    except Exception:
        return summary


# -----------------------------
# 9. MEETING PREPARATION AI
# -----------------------------
def run_gemini_preparation(prompt: str) -> str:
    """Run Gemini for meeting preparation tasks."""
    try:
        response = GEMINI_MODEL.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 2048,
            },
        )
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        
        for c in response.candidates:
            if c.content.parts:
                part = c.content.parts[0]
                if hasattr(part, "text"):
                    return part.text.strip()
        
        return str(response)
    except Exception as e:
        return f"⚠️ Gemini Error: {e}"


def generate_meeting_context(plan: MeetingPlan) -> str:
    """Generate features and broader meeting context."""
    prompt = f"""
Analyze the meeting context for {plan.company_name}, considering:
1. Meeting objective: {plan.objective}
2. Attendees: {plan.attendees}
3. Duration: {plan.duration} minutes
4. Focus areas: {plan.focus_areas}


 Relevant competitors’ feature comparisons if applicable

Format in clean, structured markdown with clear headings.
"""
    return run_gemini_preparation(prompt)


def generate_meeting_strategy(plan: MeetingPlan) -> str:
    """Generate tech stack and agenda for the meeting."""
    prompt = f"""
Develop a detailed tech stack overview and meeting agenda for the {plan.duration}-minute meeting with {plan.company_name}.

Meeting Title: {plan.title}
Objective: {plan.objective}
Attendees: {plan.attendees}
Focus Areas: {plan.focus_areas}

Include:
1. Recommended tech stack choices relevant to the objective
2. A time-boxed agenda with clear objectives for each section

Format in clean, structured markdown with time allocations and bullet points.
"""
    return run_gemini_preparation(prompt)


def generate_executive_brief(plan: MeetingPlan) -> str:
    """Generate task and deadline discussion summary."""
    prompt = f"""
Create a task-focused executive brief for the meeting with {plan.company_name}.

Meeting Details:
- Title: {plan.title}
- Objective: {plan.objective}
- Attendees: {plan.attendees}
- Duration: {plan.duration} minutes
- Focus Areas: {plan.focus_areas}

Include:
1. A summary of key decisions expected from the meeting
2. Identification of major tasks that should be assigned
3. Clear deadlines and milestones for each task
4. Suggested owners or roles responsible for each task
5. Potential blockers and mitigation strategies
6. A timeline of next steps with priority levels

Format in clean, structured markdown that’s easy to follow.
"""
    return run_gemini_preparation(prompt)



# -----------------------------
# 10. MARKDOWN FORMATTER
# -----------------------------
def format_markdown_block(
    data: Dict[str, Any],
    followup_email: str,
    whatsapp_msg: str,
) -> str:
    """Create a formatted markdown summary block."""
    summary = data.get("summary", "No summary available.")
    action_points = data.get("action_points") or []
    tasks = data.get("tasks") or []
    deadlines = data.get("deadlines") or []

    out = ""

    # Summary
    out += "##  Meeting Summary\n\n"
    out += f"{summary}\n\n"

    # Action points
    out += "##  Action Points\n\n"
    if action_points:
        for a in action_points:
            out += f"- {a}\n"
    else:
        out += "_No action points detected._\n"
    out += "\n"

    # Tasks
    out += "##  Tasks Assigned\n\n"
    if tasks:
        for t in tasks:
            assignee = t.get("assignee") or t.get("speaker") or "Unknown"
            task_text = t.get("task") or ""
            deadline = t.get("deadline") or "No deadline"
            out += f"**{assignee}** → {task_text} _(Deadline: {deadline})_\n\n"
    else:
        out += "_No explicit tasks found._\n\n"

    # Deadlines
    out += "##  Deadlines\n\n"
    if deadlines:
        for d in deadlines:
            out += f"- {d}\n"
    else:
        out += "_None mentioned._\n"
    out += "\n"

    # Follow-up email
    out += "##  Follow-up Email\n\n"
    if followup_email and not followup_email.lower().startswith("email generation failed"):
        out += "```\n"
        out += followup_email.strip() + "\n"
        out += "```\n\n"
    else:
        out += "_Could not generate email._\n\n"

    # WhatsApp
    out += "##  WhatsApp Summary\n\n"
    if whatsapp_msg and not whatsapp_msg.lower().startswith("whatsapp summary failed"):
        out += "```\n"
        out += whatsapp_msg.strip() + "\n"
        out += "```\n"
    else:
        out += "_No WhatsApp summary generated._\n"

    return out


# -----------------------------
# 11. API ENDPOINTS
# -----------------------------

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Smart Meeting Assistant API",
        "endpoints": [
            "/prepare-meeting",
            "/summarize",
        ]
    }


@app.post("/prepare-meeting")
async def prepare_meeting(request: PreparationRequest):
    """
    Generate AI-powered meeting preparation materials.
    Returns context analysis, meeting strategy, and executive brief.
    """
    try:
        plan = request.meeting_plan
        
        # Generate preparation materials
        context = generate_meeting_context(plan)
        strategy = generate_meeting_strategy(plan)
        brief = generate_executive_brief(plan)
        
        return {
            "status": "success",
            "context_analysis": context,
            "meeting_strategy": strategy,
            "executive_brief": brief,
            "meeting_plan": {
                "company_name": plan.company_name,
                "title": plan.title,
                "objective": plan.objective,
                "attendees": plan.attendees,
                "duration": plan.duration,
                "focus_areas": plan.focus_areas,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preparation failed: {str(e)}")


@app.post("/summarize")
async def summarize_meeting(
    audio: UploadFile = File(...),
    meeting_title: Optional[str] = Form(None),
    target_lang: Optional[str] = Form(None),
):
    # Save uploaded file to temp
    try:
        tmp_path = Path(tempfile.gettempdir()) / f"up_{audio.filename}"
        with open(tmp_path, "wb") as f:
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        print("❌ Failed saving upload:", e)
        raise HTTPException(status_code=500, detail=f"Upload save failed: {e}")

    if not tmp_path.exists():
        raise HTTPException(status_code=400, detail="Failed to save uploaded file.")

    try:
        # convert to wav
        try:
            wav_path = convert_to_wav(tmp_path)
        except Exception as e:
            print("❌ convert_to_wav error:", e)
            raise HTTPException(status_code=500, detail=f"Audio conversion failed: {e}")

        # transcribe
        try:
            transcript, diarization = transcribe_audio(wav_path)
        except Exception as e:
            print("❌ transcribe_audio error:", e)
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

        if not transcript.strip():
            raise HTTPException(status_code=400, detail="Transcription is empty.")

        # structured extraction
        try:
            extracted = gemini_extract_structured(transcript)
        except Exception as e:
            print("⚠️ Gemini structured extract failed:", e)
            extracted = local_extract(transcript)

        # generate email/whatsapp
        try:
            summary_text = extracted.get("summary", "")
            tasks = extracted.get("tasks", [])
            followup_email = gemini_generate_followup_email(summary_text, tasks, meeting_title or "")
            whatsapp_msg = gemini_generate_whatsapp(summary_text)
        except Exception as e:
            print("⚠️ Email/WhatsApp generation failed:", e)
            followup_email = ""
            whatsapp_msg = ""

        translated_summary = None
        if target_lang:
            try:
                translated_summary = gemini_translate_summary(summary_text, target_lang)
            except Exception as e:
                print("⚠️ Translation failed:", e)
                translated_summary = None

        structured_markdown = format_markdown_block(
            extracted,
            followup_email=followup_email,
            whatsapp_msg=whatsapp_msg,
        )

        return {
            "status": "success",
            "transcript": transcript,
            "diarization": diarization,
            "summary": summary_text,
            "action_points": extracted.get("action_points", []),
            "tasks": tasks,
            "deadlines": extracted.get("deadlines", []),
            "speakers": extracted.get("speakers", []),
            "structured_summary": structured_markdown,
            "followup_email": followup_email,
            "whatsapp": whatsapp_msg,
            "translated_summary": translated_summary,
            "target_lang": target_lang,
        }

    finally:
        # cleanup
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            if "wav_path" in locals() and wav_path.exists():
                wav_path.unlink()
        except Exception as e:
            print("⚠️ Cleanup error:", e)



# -----------------------------
# 12. RUN SERVER
# -----------------------------


    print("=" * 60)
    print(" Smart Meeting Assistant Backend")
    print("=" * 60)
    print(f" Server: http://0.0.0.0:8000")
    print(f" Docs: http://0.0.0.0:8000/docs")
    print(f" AI Model: {GEMINI_MODEL_NAME}")
    print("=" * 60)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "combined_backend:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

