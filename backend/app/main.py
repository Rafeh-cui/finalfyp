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
# Text normalisation helpers
# ---------------------------------------------------------------------------

# Question words to strip so "what is the fee" ≈ "fee"
_QW_EN = re.compile(
    r"\b(what|whats|what's|who|when|where|how|which|why|is|are|was|were|"
    r"do|does|did|can|could|will|would|should|has|have|had|the|a|an|of|"
    r"for|in|at|to|by|with|about|any|some|tell|me|give|please|"
    r"much|many|long|often|far|old|new|more|less|"
    r"i|my|you|your|we|our|their|his|her|us|it|its|this|that|these|those|"
    r"and|or|but|on|from|into|after|before|during|between|through|"
    r"use|using|used|get|getting|find|know|need|want|like|see|make|"
    r"available|provide|provided|offer|offered)\b",
    re.IGNORECASE,
)

# Urdu question / stop words to strip
_QW_UR = re.compile(
    r"(کیا|کب|کہاں|کیسے|کون|کتنا|کتنی|کتنے|کے|کا|کی|ہے|ہیں|میں|کو|"
    r"سے|پر|اور|یا|لیے|یہ|وہ|نہیں|بھی|تو|ہی|جو|جس|جن|"
    r"مجھے|آپ|ہم|آپ کا|آپ کی|ان|اس|وہاں|یہاں|جب|جہاں)",
    re.UNICODE,
)

# Common synonym / variant mapping  (key → canonical term fed to the index)
_SYNONYMS: dict[str, str] = {
    "fees": "fee",
    "tuition": "fee",
    "fee structure": "fee structure",
    "scholarships": "scholarship",
    "scholarship": "scholarship",
    "financial aid": "scholarship",
    "apply": "admission",
    "application": "admission",
    "applications": "admission",
    "admissions": "admission",
    "enroll": "admission",
    "enrolment": "admission",
    "eligibility": "admission",
    "requirements": "admission",
    "criteria": "admission",
    "hostel": "accommodation",
    "hostels": "accommodation",
    "dormitory": "accommodation",
    "transport": "transport",
    "bus": "transport",
    "transportation": "transport",
    "conveyance": "transport",
    "contact": "contact",
    "phone": "contact",
    "email": "contact",
    "address": "contact",
    "location": "contact",
    "whatsapp": "contact",
    "faculty": "faculty",
    "professor": "faculty",
    "teacher": "faculty",
    "lecturer": "faculty",
    "staff": "faculty",
    "hod": "faculty",
    "programs": "program",
    "programme": "program",
    "programmes": "program",
    "degree": "program",
    "degrees": "program",
    "courses": "course",
    "grading": "grade",
    "gpa": "grade",
    "cgpa": "grade",
    "grades": "grade",
    "result": "grade",
    "results": "grade",
    "exam": "examination",
    "exams": "examination",
    "test": "examination",
    "tests": "examination",
    "midterm": "examination",
    "final": "examination",
    "library": "library",
    "lab": "laboratory",
    "labs": "laboratory",
    "laboratories": "laboratory",
    "laboratory": "laboratory",
    "computer science": "cs",
    "electrical engineering": "ee",
    "computer engineering": "ce",
    "management sciences": "management",
    "management science": "management",
    "mathematics": "math",
    "maths": "math",
    "bs": "undergraduate",
    "bscs": "undergraduate cs",
    "bachelor": "undergraduate",
    "bachelors": "undergraduate",
    "ms": "graduate",
    "masters": "graduate",
    "master": "graduate",
    "phd": "doctoral",
    "doctorate": "doctoral",
    "research": "research",
    "thesis": "research",
    "dissertation": "research",
    "society": "society",
    "societies": "society",
    "club": "society",
    "clubs": "society",
    "events": "event",
    "activities": "event",
}


def _normalize(text: str) -> str:
    """Lowercase, strip question words, expand synonyms, clean punctuation."""
    t = text.lower().strip()
    # Remove punctuation except Urdu letters
    t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t)
    # Strip English question/stop words
    t = _QW_EN.sub(" ", t)
    # Strip Urdu question/stop words
    t = _QW_UR.sub(" ", t)
    # Apply synonym map (word-level)
    words = t.split()
    mapped = []
    i = 0
    while i < len(words):
        # Try 2-word phrases first
        if i + 1 < len(words):
            bigram = words[i] + " " + words[i + 1]
            if bigram in _SYNONYMS:
                mapped.append(_SYNONYMS[bigram])
                i += 2
                continue
        word = words[i]
        mapped.append(_SYNONYMS.get(word, word))
        i += 1
    t = " ".join(mapped)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Dataset Knowledge Base — BM25 + TF-IDF hybrid
# ---------------------------------------------------------------------------

GREETINGS = {
    "hi", "hello", "hey", "salam", "assalam", "assalamualaikum", "aoa",
    "good morning", "good afternoon", "good evening", "good day",
    "how are you", "what's up", "howdy", "greetings", "sup",
}


def _is_greeting(text: str) -> bool:
    t = text.lower().strip().rstrip("!?.،")
    return t in GREETINGS or any(t.startswith(g + " ") for g in GREETINGS)


class DatasetKnowledgeBase:
    """Triple-scored knowledge base: BM25 + word TF-IDF + char TF-IDF.

    Scoring weights: BM25 (35%) + word TF-IDF (45%) + char TF-IDF (20%).
    Queries are normalised before matching to improve recall.
    """

    def __init__(self, dataset_path: str):
        self.qa_pairs: list[dict] = []
        self.questions: list[str] = []
        self.norm_questions: list[str] = []
        self._bm25 = None
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
            self.norm_questions = [_normalize(q) for q in self.questions]
            self._build_index()
            print(f"[EduBot] Dataset loaded: {len(self.qa_pairs)} Q&A pairs.")
        except Exception as e:
            print(f"[EduBot] WARNING: Failed to load dataset ({e}). Dataset fallback disabled.")

    def _build_index(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from rank_bm25 import BM25Okapi

            # BM25 on normalised tokenised questions
            tokenized = [q.split() for q in self.norm_questions]
            self._bm25 = BM25Okapi(tokenized)

            # Char n-gram TF-IDF (surface similarity)
            self._char_vec = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4),
                max_features=10000, sublinear_tf=True,
            )
            self._char_mat = self._char_vec.fit_transform(self.norm_questions)

            # Word n-gram TF-IDF (semantic/topical similarity)
            self._word_vec = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 3),
                max_features=10000, sublinear_tf=True,
            )
            self._word_mat = self._word_vec.fit_transform(self.norm_questions)

            self._ready = True
        except Exception as e:
            print(f"[EduBot] WARNING: Index build failed ({e}).")

    def _scores(self, query: str):
        """Return blended similarity scores for all Q&A pairs."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        norm_q = _normalize(query)
        tokens = norm_q.split() if norm_q.strip() else query.lower().split()

        # BM25 — normalise to [0,1]
        bm25_raw = np.array(self._bm25.get_scores(tokens), dtype=float)
        bm25_max = bm25_raw.max()
        bm25 = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw

        # Char TF-IDF cosine
        char_q = self._char_vec.transform([norm_q])
        char = cosine_similarity(char_q, self._char_mat).flatten()

        # Word TF-IDF cosine
        word_q = self._word_vec.transform([norm_q])
        word = cosine_similarity(word_q, self._word_mat).flatten()

        return 0.35 * bm25 + 0.45 * word + 0.20 * char

    def find_similar(self, query: str, top_k: int = 5, min_score: float = 0.15) -> list[dict]:
        """Return top_k Q&A pairs above min_score threshold."""
        if not self._ready or not query.strip():
            return []
        try:
            import numpy as np
            scores = self._scores(query)
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_indices:
                score = float(scores[idx])
                if score >= min_score:
                    results.append({
                        "input": self.qa_pairs[idx]["input"],
                        "output": self.qa_pairs[idx]["output"],
                        "score": round(score, 4),
                    })
            return results
        except Exception:
            return []

    def best_answer(self, query: str, min_score: float = 0.45) -> Optional[str]:
        """Return direct dataset answer only when confidence is high."""
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

SYSTEM_PROMPT_TEMPLATE = """You are EduBot — the official AI assistant for COMSATS University Islamabad, Attock Campus.

STRICT RULES:
1. LANGUAGE: If the student writes in Urdu → reply in Urdu. If English → reply in English. Never mix.
2. ACCURACY: Only answer from the KNOWLEDGE BASE below. Do NOT invent facts, fees, names, or dates.
3. FORMAT: Keep answers concise and student-friendly. Use bullet points for lists. Maximum 5 sentences.
4. SCOPE: Only answer questions about COMSATS Attock Campus. Politely decline unrelated questions.
5. FALLBACK: If the Knowledge Base has no answer, say so and direct to: 057-9316330 or [email protected]

You help with:
- Admissions, eligibility, how to apply
- Fee structure, scholarships, financial aid
- Programs: BS / MS / PhD in CS, EE, CE, Management, Mathematics
- Faculty, departments, HoDs
- Campus life, library, labs, hostel, transport
- Exams, grades, CU-Online portal, academic calendar
- Mental health, counseling, student support

{dataset_context}"""

DATASET_CONTEXT_TEMPLATE = """
=== KNOWLEDGE BASE (answer from this — highest priority) ===
{qa_pairs}
=== END KNOWLEDGE BASE ===

Instructions: The KNOWLEDGE BASE above contains verified Q&A pairs. Use the most relevant pair(s) to construct your answer. Expand naturally but DO NOT contradict or add information not present in the knowledge base.
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
    """Build system prompt enriched with top relevant dataset context."""
    if _is_greeting(question):
        return SYSTEM_PROMPT_TEMPLATE.format(dataset_context="")

    # Fetch top 6 most relevant pairs (BM25 + TF-IDF blended)
    similar = kb.find_similar(question, top_k=6, min_score=0.12)
    if similar:
        qa_text = "\n\n".join(
            f"Q: {item['input']}\nA: {item['output']}"
            for item in similar
        )
        context = DATASET_CONTEXT_TEMPLATE.format(qa_pairs=qa_text)
    else:
        context = "\n(No matching knowledge base entries found. Answer from general COMSATS Attock knowledge only.)"
    return SYSTEM_PROMPT_TEMPLATE.format(dataset_context=context)


def _call_gemini(question: str, model_name: str) -> str:
    from google.genai import types as _types
    system_prompt = _build_system_prompt(question)
    response = gemini_client.models.generate_content(
        model=model_name,
        contents=question,
        config=_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.4,        # lower = more factual, less hallucination
            max_output_tokens=700,
        ),
    )
    text = response.text.strip() if response.text else ""
    return text or "I'm sorry, I couldn't generate a response. Please try rephrasing your question."


def _dataset_answer(question: str) -> str:
    """Smart dataset fallback with confidence-based answer selection."""
    lang = _detect_language(question)

    # ── Greetings ────────────────────────────────────────────────────────────
    if _is_greeting(question):
        if lang == "ur":
            return "السلام علیکم! میں EduBot ہوں — COMSATS اٹک کیمپس کا آفیشل معاون۔ آپ کا کیا سوال ہے؟"
        return "Hello! I'm EduBot, your official assistant for COMSATS Attock Campus. How can I help you today?"

    similar = kb.find_similar(question, top_k=5, min_score=0.12)

    if not similar:
        if lang == "ur":
            return ("معذرت، اس سوال کا جواب میرے ڈیٹا بیس میں نہیں ہے۔ "
                    "براہ کرم 057-9316330 پر کال کریں یا [email protected] پر ای میل کریں۔")
        return ("I'm sorry, I couldn't find an answer to that question. "
                "Please contact COMSATS Attock Campus at 057-9316330 or email [email protected].")

    best = similar[0]
    second = similar[1] if len(similar) > 1 else None

    # ── Very high confidence → direct answer ─────────────────────────────────
    if best["score"] >= 0.50:
        return best["output"]

    # ── High confidence, clear winner ────────────────────────────────────────
    if best["score"] >= 0.35:
        if second is None or best["score"] > second["score"] * 1.25:
            return best["output"]
        # Two strong matches → combine
        if lang == "ur":
            intro = "آپ کے سوال سے متعلق یہ معلومات ملی ہیں:"
        else:
            intro = "Here's what I found related to your question:"
        return intro + "\n\n" + "\n\n".join(f"• {r['output']}" for r in similar[:2])

    # ── Medium confidence (0.20–0.35) — use best if it clearly leads ─────────
    if best["score"] >= 0.20:
        if second is None or best["score"] > second["score"] * 1.30:
            return best["output"]
        # Ambiguous — offer top result with a caveat
        if lang == "ur":
            caveat = "شاید آپ یہ جاننا چاہتے ہیں:\n\n"
        else:
            caveat = "You might be asking about:\n\n"
        return caveat + best["output"]

    # ── Low confidence fallback ───────────────────────────────────────────────
    if lang == "ur":
        return ("معذرت، مجھے یقینی جواب نہیں مل سکا۔ "
                "براہ کرم 057-9316330 پر کال کریں یا [email protected] پر ای میل کریں۔")
    return ("I'm sorry, I couldn't find a confident answer to that question. "
            "Please contact COMSATS Attock Campus at 057-9316330 or email [email protected].")


def _generate_response(question: str) -> tuple[str, str, Optional[str]]:
    """Try Gemini first (with enriched dataset context), fall back to dataset search."""
    transient_codes = [
        "429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE",
        "quota", "rate", "overload", "capacity",
    ]

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
                print(f"[EduBot] Gemini error ({model_name}): {e}")
                break

        print("[EduBot] Gemini unavailable — falling back to dataset.")

    answer = _dataset_answer(question)
    language = _detect_language(answer)
    audio_base64 = _synthesize_audio(answer, language)
    print("[EduBot] Answered via dataset fallback.")
    return answer, language, audio_base64


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="EduBot Backend", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse


class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response: StarletteResponse = await call_next(request)
        if request.url.path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response


app.add_middleware(NoCacheMiddleware)


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
            timeout=25.0,
        )
    except asyncio.TimeoutError:
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
