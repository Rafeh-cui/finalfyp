from __future__ import annotations

import asyncio
import base64
import io
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
gemini_model = None

try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT,
        )
        GEMINI_AVAILABLE = True
        print("[EduBot] Gemini AI model loaded successfully.")
    else:
        print("[EduBot] WARNING: GEMINI_API_KEY not set. Running in demo mode.")
except ImportError:
    print("[EduBot] WARNING: google-generativeai not installed. Running in demo mode.")
except Exception as e:
    print(f"[EduBot] WARNING: Failed to init Gemini ({e}). Running in demo mode.")


# ---------------------------------------------------------------------------
# TTS helper
# ---------------------------------------------------------------------------

def _synthesize_audio(text: str, language: Optional[str] = "en") -> Optional[str]:
    if not text:
        return None
    lang = language or "en"
    # Only support English/Urdu for gTTS
    supported = {"en", "ur"}
    if lang not in supported:
        lang = "en"
    buffer = io.BytesIO()
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(buffer)
        audio_bytes = buffer.getvalue()
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception:
        return None


def _detect_language(text: str) -> str:
    try:
        from langdetect import detect, LangDetectException
        return detect(text)
    except Exception:
        return "en"


def _generate_gemini_response(question: str) -> tuple[str, str, Optional[str]]:
    if not GEMINI_AVAILABLE or gemini_model is None:
        answer = (
            "I'm EduBot, your campus assistant! The AI model is currently in demo mode. "
            "To enable full AI responses, please configure the GEMINI_API_KEY. "
            "For now, please contact the COMSATS Attock Campus admin office or visit "
            "edubotofficial@gmail.com for assistance."
        )
        return answer, "en", _synthesize_audio(answer, "en")

    response = gemini_model.generate_content(question)
    answer = response.text.strip()
    language = _detect_language(answer)
    audio_base64 = _synthesize_audio(answer, language)
    return answer, language, audio_base64


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="EduBot Backend", version="2.0.0")

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


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    loop = asyncio.get_event_loop()
    answer, language, audio_base64 = await loop.run_in_executor(
        None, _generate_gemini_response, question
    )

    if not answer:
        raise HTTPException(status_code=500, detail="Failed to generate a response. Please try again.")

    return ChatResponse(answer=answer, language=language, audio_base64=audio_base64)
