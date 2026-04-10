from __future__ import annotations

import asyncio
import base64
import io
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

SYSTEM_PROMPT = """You are EduBot, the official AI assistant for COMSATS University Islamabad, Attock Campus.
Your role is to help students with:
- Academic information: courses, schedules, exam timetables, assignments, grading criteria
- Administrative support: admissions, fee structure, scholarships, faculty contacts, office hours
- Graduate program guidance: MS/PhD requirements, thesis submission, supervisor availability, research opportunities
- Mental health & wellness: stress management tips, motivational support, counseling resources
- Campus life: events, societies, facilities, campus navigation
- General university information specific to COMSATS Attock Campus

Always provide accurate, concise, and student-friendly answers. If a question is outside your knowledge, ask for clarification or direct the student to the appropriate campus channel (admin office, department office, or faculty). Do not invent facts. Be supportive, friendly, and encouraging."""

GEMINI_AVAILABLE = False
gemini_client = None

try:
    from google import genai
    from google.genai import types as genai_types

    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
        print("[EduBot] Gemini AI client loaded successfully (google-genai SDK).")
    else:
        print("[EduBot] WARNING: GEMINI_API_KEY not set. Running in demo mode.")
except ImportError:
    print("[EduBot] WARNING: google-genai not installed. Running in demo mode.")
except Exception as e:
    print(f"[EduBot] WARNING: Failed to init Gemini ({e}). Running in demo mode.")


# ---------------------------------------------------------------------------
# TTS helper
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> str:
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"


def _synthesize_audio(text: str, language: Optional[str] = "en") -> Optional[str]:
    if not text:
        return None
    lang = language or "en"
    if lang not in {"en", "ur"}:
        lang = "en"
    buffer = io.BytesIO()
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(buffer)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        return None


def _generate_gemini_response(question: str) -> tuple[str, str, Optional[str]]:
    if not GEMINI_AVAILABLE or gemini_client is None:
        answer = (
            "I'm EduBot, your campus assistant! The AI model is currently in demo mode. "
            "To enable full AI responses, please configure the GEMINI_API_KEY. "
            "For now, please contact the COMSATS Attock Campus admin office or visit "
            "edubotofficial@gmail.com for assistance."
        )
        return answer, "en", _synthesize_audio(answer, "en")

    from google.genai import types as genai_types

    models_to_try = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]

    def _try_model(model_name: str) -> str:
        from google.genai import types as _types
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=question,
            config=_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=512,
            ),
        )
        text = response.text.strip() if response.text else ""
        return text or "I'm sorry, I couldn't generate a response. Please try rephrasing your question."

    for model_name in models_to_try:
        try:
            answer = _try_model(model_name)
            language = _detect_language(answer)
            audio_base64 = _synthesize_audio(answer, language)
            return answer, language, audio_base64
        except Exception as e:
            error_str = str(e)
            transient = ["429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "quota", "rate"]
            if any(code in error_str for code in transient):
                continue
            raise

    answer = (
        "I'm currently experiencing high traffic. Please try again in a moment, "
        "or visit edubotofficial@gmail.com for assistance."
    )
    return answer, "en", _synthesize_audio(answer, "en")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="EduBot Backend", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    language: Optional[str] = None
    audio_base64: Optional[str] = None


@app.get("/health", tags=["System"])
async def health_check() -> dict:
    return {
        "status": "ok",
        "ai_engine": "gemini-2.0-flash" if GEMINI_AVAILABLE else "demo-mode",
        "gemini_available": GEMINI_AVAILABLE,
    }


@app.get("/", tags=["Frontend"])
async def root():
    return RedirectResponse(url="/firstscreen.html")


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    loop = asyncio.get_event_loop()
    try:
        answer, language, audio_base64 = await asyncio.wait_for(
            loop.run_in_executor(None, _generate_gemini_response, question),
            timeout=20.0
        )
    except asyncio.TimeoutError:
        answer = (
            "I'm taking too long to respond right now. Please try again in a moment, "
            "or contact edubotofficial@gmail.com for assistance."
        )
        language = "en"
        audio_base64 = None

    return ChatResponse(answer=answer, language=language, audio_base64=audio_base64)


# Serve frontend static files — must be mounted LAST so API routes take priority
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
_frontend_dir = os.path.realpath(_frontend_dir)
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
    print(f"[EduBot] Serving frontend static files from: {_frontend_dir}")
