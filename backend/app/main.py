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

GREETINGS = {
    "hi", "hello", "hey", "salam", "assalam", "assalamualaikum", "good morning",
    "good afternoon", "good evening", "good day", "how are you", "what's up",
    "howdy", "greetings", "sup",
}

def _is_greeting(text: str) -> bool:
    t = text.lower().strip().rstrip("!?.،")
    return t in GREETINGS or any(t.startswith(g + " ") for g in GREETINGS)


class DatasetKnowledgeBase:
    """Loads the COMSATS Q&A dataset and finds similar answers using a hybrid
    char n-gram + word n-gram TF-IDF scorer for higher precision."""

    def __init__(self, dataset_path: str):
        self.qa_pairs: list[dict] = []
        self.questions: list[str] = []
        self._char_vec = None
        self._char_mat = None
        self._word_vec = None
        self._word_mat = None
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
            # Character n-gram scorer (surface similarity)
            self._char_vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                max_features=8000,
                sublinear_tf=True,
            )
            self._char_mat = self._char_vec.fit_transform(self.questions)
            # Word n-gram scorer (semantic/topical similarity)
            self._word_vec = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                max_features=8000,
                sublinear_tf=True,
            )
            self._word_mat = self._word_vec.fit_transform(self.questions)
            self._ready = True
        except Exception as e:
            print(f"[EduBot] WARNING: TF-IDF index build failed ({e}).")

    def _hybrid_scores(self, query: str):
        """Return combined (char*0.4 + word*0.6) similarity scores as numpy array."""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        char_scores = cosine_similarity(
            self._char_vec.transform([query]), self._char_mat
        ).flatten()
        word_scores = cosine_similarity(
            self._word_vec.transform([query]), self._word_mat
        ).flatten()
        return 0.4 * char_scores + 0.6 * word_scores

    def find_similar(self, query: str, top_k: int = 5, min_score: float = 0.20) -> list[dict]:
        """Return top_k Q&A pairs most similar to query, filtered by min_score."""
        if not self._ready or not query.strip():
            return []
        try:
            import numpy as np
            scores = self._hybrid_scores(query)
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

    def best_answer(self, query: str, min_score: float = 0.50) -> Optional[str]:
        """Return the best dataset answer only when confidence is high enough."""
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
    # Don't inject noisy context for greetings
    if _is_greeting(question):
        return SYSTEM_PROMPT_TEMPLATE.format(dataset_context="")
    similar = kb.find_similar(question, top_k=5, min_score=0.20)
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
    lang = _detect_language(question)

    # Greetings — respond naturally without a dataset lookup
    if _is_greeting(question):
        if lang == "ur":
            return "السلام علیکم! میں EduBot ہوں، COMSATS اٹک کیمپس کا ذہین معاون۔ آپ کا کیا سوال ہے؟"
        return "Hello! I'm EduBot, your smart assistant for COMSATS Attock Campus. How can I help you today?"

    # High-confidence direct answer (hybrid score ≥ 0.50)
    answer = kb.best_answer(question, min_score=0.50)
    if answer:
        return answer

    # Medium-confidence match — only use it when there's a clear winner
    similar = kb.find_similar(question, top_k=3, min_score=0.25)
    if similar:
        best = similar[0]
        # Use top result only if it clearly leads the pack (20% margin) and score ≥ 0.42
        if best["score"] >= 0.42:
            if len(similar) == 1 or best["score"] > similar[1]["score"] * 1.2:
                return best["output"]
            # Multiple plausible results — combine top 2
            if lang == "ur":
                intro = "آپ کے سوال سے متعلق یہ معلومات ہیں:"
            else:
                intro = "Here's what I found related to your question:"
            combined = "\n\n".join(f"• {r['output']}" for r in similar[:2])
            return f"{intro}\n\n{combined}"

    # No confident match found
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
