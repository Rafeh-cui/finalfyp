from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Dataset Knowledge Base
# ---------------------------------------------------------------------------

class DatasetKnowledgeBase:
    """Loads the COMSATS Q&A dataset and finds similar answers via TF-IDF."""

    def __init__(self, dataset_path: str):
        self.qa_pairs: list[dict] = []
        self.questions: list[str] = []
        self._vectorizer = None
        self._matrix = None
        self._ready = False
        self._load(dataset_path)

    def _load(self, path: str):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.qa_pairs = [
                {"input": d["input"].strip(), "output": d["output"].strip()}
                for d in data
                if d.get("input") and d.get("output")
            ]
            self.questions = [d["input"] for d in self.qa_pairs]
            self._build_index()
            print(f"[EduBot] Dataset loaded: {len(self.qa_pairs)} Q&A pairs.")
        except Exception as e:
            print(f"[EduBot] WARNING: Failed to load dataset ({e}). Dataset fallback disabled.")

    def _build_index(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                max_features=8000,
                sublinear_tf=True,
            )
            self._matrix = self._vectorizer.fit_transform(self.questions)
            self._ready = True
        except Exception as e:
            print(f"[EduBot] WARNING: TF-IDF index build failed ({e}).")

    def find_similar(self, query: str, top_k: int = 5, min_score: float = 0.15) -> list[dict]:
        """Return top_k Q&A pairs most similar to query, filtered by min_score."""
        if not self._ready or not query.strip():
            return []
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            q_vec = self._vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self._matrix).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_indices:
                score = float(scores[idx])
                if score >= min_score:
                    results.append({
                        "input": self.qa_pairs[idx]["input"],
                        "output": self.qa_pairs[idx]["output"],
                        "score": score,
                    })
            return results
        except Exception:
            return []

    def best_answer(self, query: str, min_score: float = 0.30) -> Optional[str]:
        """Return the best dataset answer if similarity is high enough, else None."""
        results = self.find_similar(query, top_k=1, min_score=min_score)
        if results:
            return results[0]["output"]
        return None


_dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
kb = DatasetKnowledgeBase(_dataset_path)


# ---------------------------------------------------------------------------
# Configuration & Gemini setup
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

SYSTEM_PROMPT_TEMPLATE = """You are EduBot, the official AI assistant for COMSATS University Islamabad, Attock Campus.

Your role is to help students with:
- Academic information: courses, schedules, exam timetables, assignments, grading criteria
- Administrative support: admissions, fee structure, scholarships, faculty contacts, office hours
- Graduate program guidance: MS/PhD requirements, thesis submission, supervisor availability, research opportunities
- Mental health & wellness: stress management tips, motivational support, counseling resources
- Campus life: events, societies, facilities, campus navigation
- General university information specific to COMSATS Attock Campus

LANGUAGE RULE: If the student's question is in Urdu, respond in Urdu. If in English, respond in English.

Always provide accurate, concise, and student-friendly answers. If a question is outside your knowledge, ask for clarification or direct the student to the appropriate campus channel (admin office, department office, or faculty). Do not invent facts. Be supportive, friendly, and encouraging.

{dataset_context}"""

DATASET_CONTEXT_BLOCK = """
RELEVANT KNOWLEDGE BASE (use these Q&A pairs to inform your answer — prioritize this information):
{qa_pairs}
"""

GEMINI_AVAILABLE = False
gemini_client = None

try:
    from google import genai
    from google.genai import types as genai_types

    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
        print("[EduBot] Gemini AI client loaded successfully.")
    else:
        print("[EduBot] WARNING: GEMINI_API_KEY not set. Running in dataset-only mode.")
except ImportError:
    print("[EduBot] WARNING: google-genai not installed. Running in dataset-only mode.")
except Exception as e:
    print(f"[EduBot] WARNING: Failed to init Gemini ({e}). Running in dataset-only mode.")


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


# ---------------------------------------------------------------------------
# Core response generation
# ---------------------------------------------------------------------------

def _build_system_prompt(question: str) -> str:
    """Build system prompt enriched with relevant dataset context."""
    similar = kb.find_similar(question, top_k=5, min_score=0.10)
    if similar:
        qa_text = "\n".join(
            f"Q: {item['input']}\nA: {item['output']}"
            for item in similar
        )
        context = DATASET_CONTEXT_BLOCK.format(qa_pairs=qa_text)
    else:
        context = ""
    return SYSTEM_PROMPT_TEMPLATE.format(dataset_context=context)


def _call_gemini(question: str, model_name: str) -> str:
    from google.genai import types as _types
    system_prompt = _build_system_prompt(question)
    response = gemini_client.models.generate_content(
        model=model_name,
        contents=question,
        config=_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.6,
            max_output_tokens=600,
        ),
    )
    text = response.text.strip() if response.text else ""
    return text or "I'm sorry, I couldn't generate a response. Please try rephrasing your question."


def _dataset_answer(question: str) -> str:
    """Search dataset for a good answer, with a conversational wrapper."""
    # Try high-confidence match first
    answer = kb.best_answer(question, min_score=0.35)
    if answer:
        return answer

    # Try looser match and build a response from top results
    similar = kb.find_similar(question, top_k=3, min_score=0.15)
    if similar:
        best = similar[0]
        if len(similar) == 1 or best["score"] > similar[1]["score"] * 1.3:
            return best["output"]
        # Multiple similar results — combine them
        lang = _detect_language(question)
        if lang == "ur":
            intro = "آپ کے سوال سے متعلق یہ معلومات ہیں:"
        else:
            intro = "Here's what I found related to your question:"
        combined = "\n\n".join(f"• {r['output']}" for r in similar[:2])
        return f"{intro}\n\n{combined}"

    # No match found
    lang = _detect_language(question)
    if lang == "ur":
        return (
            "معذرت، اس سوال کا جواب میرے ڈیٹا بیس میں موجود نہیں ہے۔ "
            "براہ کرم COMSATS اٹک کیمپس کے ایڈمن آفس یا edubotofficial@gmail.com سے رابطہ کریں۔"
        )
    return (
        "I'm sorry, I couldn't find a specific answer to that question in my knowledge base. "
        "Please contact the COMSATS Attock Campus admin office or email edubotofficial@gmail.com for assistance."
    )


def _generate_response(question: str) -> tuple[str, str, Optional[str]]:
    """
    Try Gemini first (with dataset context injected), then fall back to dataset search.
    """
    transient_codes = ["429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "quota", "rate", "overload"]

    if GEMINI_AVAILABLE and gemini_client:
        models_to_try = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]
        for model_name in models_to_try:
            try:
                answer = _call_gemini(question, model_name)
                language = _detect_language(answer)
                audio_base64 = _synthesize_audio(answer, language)
                print(f"[EduBot] Answered via Gemini ({model_name})")
                return answer, language, audio_base64
            except Exception as e:
                error_str = str(e)
                if any(code in error_str for code in transient_codes):
                    continue
                # Non-transient error — fall through to dataset
                print(f"[EduBot] Gemini error ({model_name}): {e}")
                break

        print("[EduBot] Gemini unavailable — falling back to dataset search.")

    # Dataset fallback
    answer = _dataset_answer(question)
    language = _detect_language(answer)
    audio_base64 = _synthesize_audio(answer, language)
    print("[EduBot] Answered via dataset fallback.")
    return answer, language, audio_base64


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="EduBot Backend", version="3.0.0")

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
    source: Optional[str] = None


@app.get("/health", tags=["System"])
async def health_check() -> dict:
    return {
        "status": "ok",
        "gemini_available": GEMINI_AVAILABLE,
        "dataset_loaded": kb._ready,
        "dataset_size": len(kb.qa_pairs),
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
            loop.run_in_executor(None, _generate_response, question),
            timeout=20.0,
        )
    except asyncio.TimeoutError:
        # Last resort: try dataset synchronously on timeout
        answer = _dataset_answer(question)
        language = _detect_language(answer)
        audio_base64 = _synthesize_audio(answer, language)

    return ChatResponse(answer=answer, language=language, audio_base64=audio_base64)


# Serve frontend static files — mounted LAST so API routes take priority
_frontend_dir = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
)
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
    print(f"[EduBot] Serving frontend from: {_frontend_dir}")
