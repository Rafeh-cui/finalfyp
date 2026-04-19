from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

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

_QW_UR = re.compile(
    r"(کیا|کب|کہاں|کیسے|کون|کتنا|کتنی|کتنے|کے|کا|کی|ہے|ہیں|میں|کو|"
    r"سے|پر|اور|یا|لیے|یہ|وہ|نہیں|بھی|تو|ہی|جو|جس|جن|"
    r"مجھے|آپ|ہم|آپ کا|آپ کی|ان|اس|وہاں|یہاں|جب|جہاں)",
    re.UNICODE,
)

_SYNONYMS: dict[str, str] = {
    "fees": "fee", "tuition": "fee", "fee structure": "fee structure",
    "scholarships": "scholarship", "scholarship": "scholarship",
    "financial aid": "scholarship", "apply": "admission",
    "application": "admission", "applications": "admission",
    "admissions": "admission", "enroll": "admission", "enrolment": "admission",
    "eligibility": "admission", "requirements": "admission", "criteria": "admission",
    "hostel": "accommodation", "hostels": "accommodation", "dormitory": "accommodation",
    "transport": "transport", "bus": "transport", "transportation": "transport",
    "conveyance": "transport", "contact": "contact", "phone": "contact",
    "email": "contact", "address": "contact", "location": "contact",
    "whatsapp": "contact", "faculty": "faculty", "professor": "faculty",
    "teacher": "faculty", "lecturer": "faculty", "staff": "faculty", "hod": "faculty",
    "programs": "program", "programme": "program", "programmes": "program",
    "degree": "program", "degrees": "program", "courses": "course",
    "grading": "grade", "gpa": "grade", "cgpa": "grade", "grades": "grade",
    "result": "grade", "results": "grade", "exam": "examination",
    "exams": "examination", "test": "examination", "tests": "examination",
    "midterm": "examination", "final": "examination", "library": "library",
    "lab": "laboratory", "labs": "laboratory", "laboratories": "laboratory",
    "laboratory": "laboratory", "computer science": "cs", "electrical engineering": "ee",
    "computer engineering": "ce", "management sciences": "management",
    "management science": "management", "mathematics": "math", "maths": "math",
    "bs": "undergraduate", "bscs": "undergraduate cs", "bachelor": "undergraduate",
    "bachelors": "undergraduate", "ms": "graduate", "masters": "graduate",
    "master": "graduate", "phd": "doctoral", "doctorate": "doctoral",
    "research": "research", "thesis": "research", "dissertation": "research",
    "society": "society", "societies": "society", "club": "society", "clubs": "society",
    "events": "event", "activities": "event",
}


def _normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t)
    t = _QW_EN.sub(" ", t)
    t = _QW_UR.sub(" ", t)
    words = t.split()
    mapped, i = [], 0
    while i < len(words):
        if i + 1 < len(words):
            bigram = words[i] + " " + words[i + 1]
            if bigram in _SYNONYMS:
                mapped.append(_SYNONYMS[bigram])
                i += 2
                continue
        mapped.append(_SYNONYMS.get(words[i], words[i]))
        i += 1
    return re.sub(r"\s+", " ", " ".join(mapped)).strip()


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
    """Triple-scored knowledge base: BM25 (35%) + word TF-IDF (45%) + char TF-IDF (20%)."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.qa_pairs: list[dict] = []
        self.questions: list[str] = []
        self.norm_questions: list[str] = []
        self._bm25 = None
        self._char_vec = None
        self._char_mat = None
        self._word_vec = None
        self._word_mat = None
        self._ready = False
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        try:
            with open(self.dataset_path, encoding="utf-8") as f:
                data = json.load(f)
            pairs = [
                {"input": d["input"].strip(), "output": d["output"].strip()}
                for d in data
                if d.get("input") and d.get("output")
            ]
            self.qa_pairs = pairs
            self.questions = [d["input"] for d in pairs]
            self.norm_questions = [_normalize(q) for q in self.questions]
            self._build_index()
            print(f"[EduBot] Dataset loaded: {len(self.qa_pairs)} Q&A pairs.")
        except Exception as e:
            print(f"[EduBot] WARNING: Failed to load dataset ({e}).")

    def _build_index(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from rank_bm25 import BM25Okapi

            tokenized = [q.split() for q in self.norm_questions]
            self._bm25 = BM25Okapi(tokenized)

            self._char_vec = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4), max_features=10000, sublinear_tf=True)
            self._char_mat = self._char_vec.fit_transform(self.norm_questions)

            self._word_vec = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 3), max_features=10000, sublinear_tf=True)
            self._word_mat = self._word_vec.fit_transform(self.norm_questions)

            self._ready = True
        except Exception as e:
            print(f"[EduBot] WARNING: Index build failed ({e}).")

    def reload(self):
        """Hot-reload dataset from disk and rebuild index (thread-safe)."""
        with self._lock:
            self._ready = False
            self._load()
        _response_cache.clear()
        print("[EduBot] Dataset reloaded and cache cleared.")

    def add_pair(self, question: str, answer: str) -> bool:
        """Append a new Q&A pair to dataset.json and hot-reload. Returns True on success."""
        question, answer = question.strip(), answer.strip()
        if not question or not answer:
            return False
        # Deduplicate: skip if very similar question already exists
        if self._ready:
            existing = self.find_similar(question, top_k=1, min_score=0.70)
            if existing:
                return False   # Already covered

        try:
            with open(self.dataset_path, encoding="utf-8") as f:
                data = json.load(f)
            data.append({
                "input": question,
                "output": answer,
                "instruction": "You are an expert assistant for COMSATS University Islamabad, Attock Campus.",
            })
            with open(self.dataset_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.reload()
            print(f"[EduBot] New Q&A saved and dataset reloaded. Total: {len(self.qa_pairs)}")
            return True
        except Exception as e:
            print(f"[EduBot] ERROR saving Q&A: {e}")
            return False

    def _scores(self, query: str):
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        norm_q = _normalize(query)
        tokens = norm_q.split() if norm_q.strip() else query.lower().split()

        bm25_raw = np.array(self._bm25.get_scores(tokens), dtype=float)
        bm25_max = bm25_raw.max()
        bm25 = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw

        char_q = self._char_vec.transform([norm_q])
        char = cosine_similarity(char_q, self._char_mat).flatten()

        word_q = self._word_vec.transform([norm_q])
        word = cosine_similarity(word_q, self._word_mat).flatten()

        return 0.35 * bm25 + 0.45 * word + 0.20 * char

    def find_similar(self, query: str, top_k: int = 5, min_score: float = 0.15) -> list[dict]:
        if not self._ready or not query.strip():
            return []
        try:
            import numpy as np
            scores = self._scores(query)
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [
                {"input": self.qa_pairs[i]["input"],
                 "output": self.qa_pairs[i]["output"],
                 "score": round(float(scores[i]), 4)}
                for i in top_indices
                if float(scores[i]) >= min_score
            ]
        except Exception:
            return []

    def best_answer(self, query: str, min_score: float = 0.45) -> Optional[str]:
        results = self.find_similar(query, top_k=1, min_score=min_score)
        return results[0]["output"] if results else None


_dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
kb = DatasetKnowledgeBase(_dataset_path)

# ---------------------------------------------------------------------------
# Unknown-question log  (persisted to disk)
# ---------------------------------------------------------------------------

_UNKNOWN_PATH = os.path.join(os.path.dirname(__file__), "unknown_questions.json")
_unknown_lock = threading.Lock()


def _load_unknown() -> list[dict]:
    try:
        with open(_UNKNOWN_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_unknown(entry: dict) -> None:
    with _unknown_lock:
        data = _load_unknown()
        # Don't duplicate exact same question
        if any(d["question"] == entry["question"] for d in data):
            return
        data.append(entry)
        with open(_UNKNOWN_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Response cache  (LRU, max 120 entries)
# ---------------------------------------------------------------------------

_CACHE_MAX = 120
_response_cache: OrderedDict[str, tuple[str, str]] = OrderedDict()


def _cache_get(key: str) -> Optional[tuple[str, str]]:
    if key in _response_cache:
        _response_cache.move_to_end(key)
        return _response_cache[key]
    return None


def _cache_put(key: str, value: tuple[str, str]) -> None:
    _response_cache[key] = value
    _response_cache.move_to_end(key)
    if len(_response_cache) > _CACHE_MAX:
        _response_cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Configuration & Gemini setup
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ADMIN_KEY = os.environ.get("ADMIN_KEY", "edubot-admin-2025")

SYSTEM_PROMPT_TEMPLATE = """You are EduBot — the official AI assistant for COMSATS University Islamabad, Attock Campus.

STRICT RULES:
1. LANGUAGE: If the student writes in Urdu → reply in Urdu. If English → reply in English. Never mix.
2. ACCURACY: Only answer from the KNOWLEDGE BASE below. Do NOT invent facts, fees, names, or dates.
3. FORMAT: Keep answers concise and student-friendly. Use bullet points for lists. Maximum 5 sentences.
4. SCOPE: Only answer questions about COMSATS Attock Campus. Politely decline unrelated questions.
5. FALLBACK: If the Knowledge Base has no answer, say so and direct to: 057-9316330 or [email protected]
6. CONTEXT: The conversation history below shows previous turns. Use it to understand follow-up questions.

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

Instructions: Use the most relevant Q&A pair(s) above to construct your answer. Expand naturally but DO NOT contradict or add information not in the knowledge base.
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
# TTS helper  (server-side fallback — frontend prefers Web Speech API)
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
    words = text.split()
    if len(words) > 80:
        text = " ".join(words[:80]) + "…"
    buffer = io.BytesIO()
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(buffer)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        return None


# Phrases that indicate the bot doesn't know the answer (used for auto-learn logic)
_FALLBACK_PHRASES = [
    "contact comsats", "contact us", "057-9316330", "couldn't find",
    "i don't know", "i'm not sure", "please contact",
    "مجھے معلوم نہیں", "057", "رابطہ کریں",
]


def _is_fallback_answer(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in _FALLBACK_PHRASES)


# ---------------------------------------------------------------------------
# Core response generation
# ---------------------------------------------------------------------------

def _build_system_prompt(question: str) -> str:
    if _is_greeting(question):
        return SYSTEM_PROMPT_TEMPLATE.format(dataset_context="")

    similar = kb.find_similar(question, top_k=6, min_score=0.12)
    if similar:
        qa_text = "\n\n".join(f"Q: {r['input']}\nA: {r['output']}" for r in similar)
        context = DATASET_CONTEXT_TEMPLATE.format(qa_pairs=qa_text)
    else:
        context = "\n(No matching knowledge base entries found. Answer from general COMSATS Attock knowledge only.)"
    return SYSTEM_PROMPT_TEMPLATE.format(dataset_context=context)


def _call_gemini(question: str, model_name: str, history: list[dict]) -> str:
    from google.genai import types as _types

    system_prompt = _build_system_prompt(question)
    contents: list = []
    for turn in history[-6:]:
        role = "user" if turn.get("role") == "user" else "model"
        contents.append(_types.Content(role=role, parts=[_types.Part(text=turn.get("content", ""))]))
    contents.append(_types.Content(role="user", parts=[_types.Part(text=question)]))

    response = gemini_client.models.generate_content(
        model=model_name,
        contents=contents,
        config=_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.35,
            max_output_tokens=600,
        ),
    )
    text = response.text.strip() if response.text else ""
    return text or "I'm sorry, I couldn't generate a response. Please try rephrasing your question."


def _dataset_answer(question: str) -> str:
    lang = _detect_language(question)

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

    if best["score"] >= 0.50:
        return best["output"]
    if best["score"] >= 0.35:
        if second is None or best["score"] > second["score"] * 1.25:
            return best["output"]
        intro = "آپ کے سوال سے متعلق یہ معلومات ملی ہیں:" if lang == "ur" else "Here's what I found:"
        return intro + "\n\n" + "\n\n".join(f"• {r['output']}" for r in similar[:2])
    if best["score"] >= 0.20:
        if second is None or best["score"] > second["score"] * 1.30:
            return best["output"]
        caveat = "شاید آپ یہ جاننا چاہتے ہیں:\n\n" if lang == "ur" else "You might be asking about:\n\n"
        return caveat + best["output"]

    if lang == "ur":
        return ("معذرت، مجھے یقینی جواب نہیں مل سکا۔ "
                "براہ کرم 057-9316330 پر کال کریں یا [email protected] پر ای میل کریں۔")
    return ("I'm sorry, I couldn't find a confident answer. "
            "Please contact COMSATS Attock Campus at 057-9316330 or email [email protected].")


def _generate_response(
    question: str,
    history: list[dict],
    want_audio: bool,
) -> tuple[str, str, Optional[str]]:
    cache_key = _normalize(question)

    # Cache hit (stateless queries only)
    if not history:
        cached = _cache_get(cache_key)
        if cached:
            answer, language = cached
            audio = _synthesize_audio(answer, language) if want_audio else None
            print(f"[EduBot] Cache hit: {question[:60]}")
            return answer, language, audio

    # Check dataset confidence BEFORE calling Gemini
    top_matches = kb.find_similar(question, top_k=1, min_score=0.0)
    dataset_confidence = top_matches[0]["score"] if top_matches else 0.0

    transient_codes = ["429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "quota", "rate", "overload"]

    if GEMINI_AVAILABLE and gemini_client:
        models_to_try = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]
        for model_name in models_to_try:
            try:
                answer = _call_gemini(question, model_name, history)
                language = _detect_language(answer)

                # ── Auto-learn: question not well-covered in dataset ─────────
                if (dataset_confidence < 0.25
                        and not _is_greeting(question)
                        and not _is_fallback_answer(answer)
                        and len(answer) > 40):
                    saved = kb.add_pair(question, answer)
                    if saved:
                        print(f"[EduBot] Auto-learned new Q&A. Dataset now: {len(kb.qa_pairs)}")

                # ── Log if Gemini also fell back to "contact us" ─────────────
                elif _is_fallback_answer(answer) and not _is_greeting(question):
                    _save_unknown({
                        "question": question,
                        "confidence": round(dataset_confidence, 3),
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                    print(f"[EduBot] Logged unknown question: {question[:60]}")

                if not history:
                    _cache_put(cache_key, (answer, language))
                audio = _synthesize_audio(answer, language) if want_audio else None
                print(f"[EduBot] Answered via Gemini ({model_name})")
                return answer, language, audio

            except Exception as e:
                if any(code in str(e) for code in transient_codes):
                    continue
                print(f"[EduBot] Gemini error ({model_name}): {e}")
                break

        print("[EduBot] Gemini unavailable — falling back to dataset.")

    answer = _dataset_answer(question)
    language = _detect_language(answer)

    # Log unknown question in dataset-only mode too
    if _is_fallback_answer(answer) and not _is_greeting(question):
        _save_unknown({
            "question": question,
            "confidence": round(dataset_confidence, 3),
            "timestamp": datetime.utcnow().isoformat(),
        })

    if not history:
        _cache_put(cache_key, (answer, language))
    audio = _synthesize_audio(answer, language) if want_audio else None
    print("[EduBot] Answered via dataset fallback.")
    return answer, language, audio


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="EduBot Backend", version="5.1.0")

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


# ── Admin key guard ───────────────────────────────────────────────────────────
def _require_admin(x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    return True


# ── Pydantic models ───────────────────────────────────────────────────────────
class HistoryTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: List[HistoryTurn] = []
    want_audio: bool = False


class ChatResponse(BaseModel):
    answer: str
    language: Optional[str] = None
    audio_base64: Optional[str] = None
    source: Optional[str] = None


class AddQARequest(BaseModel):
    question: str
    answer: str


# ── Public endpoints ──────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check() -> dict:
    return {
        "status": "ok",
        "gemini_available": GEMINI_AVAILABLE,
        "dataset_loaded": kb._ready,
        "dataset_size": len(kb.qa_pairs),
        "cache_size": len(_response_cache),
        "unknown_questions": len(_load_unknown()),
    }


@app.get("/", tags=["Frontend"])
async def root():
    return RedirectResponse(url="/firstscreen.html")


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    history = [{"role": t.role, "content": t.content} for t in request.history]
    loop = asyncio.get_event_loop()
    try:
        answer, language, audio_base64 = await asyncio.wait_for(
            loop.run_in_executor(None, _generate_response, question, history, request.want_audio),
            timeout=25.0,
        )
    except asyncio.TimeoutError:
        answer = _dataset_answer(question)
        language = _detect_language(answer)
        audio_base64 = _synthesize_audio(answer, language) if request.want_audio else None

    return ChatResponse(answer=answer, language=language, audio_base64=audio_base64)


# ── Admin endpoints ────────────────────────────────────────────────────────────
@app.get("/api/admin/stats", tags=["Admin"])
async def admin_stats(_: bool = Depends(_require_admin)):
    return {
        "dataset_size":       len(kb.qa_pairs),
        "cache_size":         len(_response_cache),
        "unknown_count":      len(_load_unknown()),
        "gemini_available":   GEMINI_AVAILABLE,
        "dataset_ready":      kb._ready,
    }


@app.get("/api/admin/unknown", tags=["Admin"])
async def admin_unknown(_: bool = Depends(_require_admin)):
    return {"unknown_questions": _load_unknown()}


@app.post("/api/admin/add-qa", tags=["Admin"])
async def admin_add_qa(body: AddQARequest, _: bool = Depends(_require_admin)):
    loop = asyncio.get_event_loop()
    saved = await loop.run_in_executor(None, kb.add_pair, body.question, body.answer)
    if not saved:
        raise HTTPException(status_code=409, detail="A very similar question already exists or input was empty.")
    return {"success": True, "dataset_size": len(kb.qa_pairs)}


@app.post("/api/admin/reload", tags=["Admin"])
async def admin_reload(_: bool = Depends(_require_admin)):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, kb.reload)
    return {"success": True, "dataset_size": len(kb.qa_pairs)}


@app.delete("/api/admin/unknown", tags=["Admin"])
async def admin_clear_unknown(_: bool = Depends(_require_admin)):
    with _unknown_lock:
        with open(_UNKNOWN_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)
    return {"success": True}


@app.delete("/api/admin/unknown/{index}", tags=["Admin"])
async def admin_delete_unknown(index: int, _: bool = Depends(_require_admin)):
    with _unknown_lock:
        data = _load_unknown()
        if index < 0 or index >= len(data):
            raise HTTPException(status_code=404, detail="Index out of range.")
        data.pop(index)
        with open(_UNKNOWN_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return {"success": True, "remaining": len(data)}


# Serve frontend static files — mounted LAST
_frontend_dir = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
)
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
    print(f"[EduBot] Serving frontend from: {_frontend_dir}")
