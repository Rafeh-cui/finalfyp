from __future__ import annotations

import asyncio
import base64
import io
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "bloomz-finetuned-model"
GENERATION_MAX_TOKENS = 90

# ---------------------------------------------------------------------------
# Model loading (deferred – model directory may not exist)
# ---------------------------------------------------------------------------

MODEL_AVAILABLE = False
MODEL = None
TOKENIZER = None
DEVICE = None
STOPPING_CRITERIA = None

try:
    import torch
    from transformers import (
        BloomForCausalLM,
        BloomTokenizerFast,
        StoppingCriteria,
        StoppingCriteriaList,
    )
    from langdetect import DetectorFactory, LangDetectException, detect
    from gtts import gTTS

    DetectorFactory.seed = 42

    class StopAfterSecondBrace(StoppingCriteria):
        """Custom stopping criteria that halts generation after two closing braces."""

        def __init__(self) -> None:
            super().__init__()
            self.brace_count = 0

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            decoded = TOKENIZER.decode(input_ids[0], skip_special_tokens=True)
            self.brace_count = decoded.count("}")
            return self.brace_count >= 2

    if MODEL_DIR.exists():
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        TOKENIZER = BloomTokenizerFast.from_pretrained(
            MODEL_DIR.as_posix(),
            use_fast=False,
        )
        MODEL = BloomForCausalLM.from_pretrained(
            MODEL_DIR.as_posix(),
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        )
        MODEL.to(DEVICE)
        MODEL.eval()
        STOPPING_CRITERIA = StoppingCriteriaList([StopAfterSecondBrace()])
        MODEL_AVAILABLE = True
        print(f"[EduBot] Model loaded successfully on {DEVICE}")
    else:
        print(f"[EduBot] WARNING: Model directory not found at {MODEL_DIR}. "
              "Running in demo mode – /api/chat will return placeholder responses.")

except ImportError as e:
    print(f"[EduBot] WARNING: Could not import ML dependencies ({e}). "
          "Running in demo mode.")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

ANSWER_PATTERN = re.compile(
    r"(?:Answer:\s*)?\{?(.*?)(?:\}|$)", flags=re.IGNORECASE | re.DOTALL
)


def _build_prompt(question: str) -> str:
    instructions = (
        "You are EduBot, the official assistant for COMSATS University Islamabad, "
        "Attock Campus. Provide accurate, concise, and student-friendly answers "
        "focused on campus academics, student life, and support services. If a "
        "question falls outside your knowledge, ask for clarification or direct "
        "the student to official campus channels. Do not invent facts."
    )
    return f"{instructions}\nQuestion: {question}\nAnswer:"


def _clean_response(raw_text: str) -> str:
    if "Answer:" in raw_text:
        text = raw_text.split("Answer:", 1)[1]
    else:
        text = raw_text

    match = ANSWER_PATTERN.search(text)
    if match:
        text = match.group(1)

    text = text.strip("{}").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _detect_language(text: str) -> Optional[str]:
    if not text:
        return None
    try:
        from langdetect import detect, LangDetectException
        return detect(text)
    except Exception:
        return None


def _synthesize_audio(text: str, language: Optional[str]) -> Optional[str]:
    if not text:
        return None

    lang = language or "en"
    buffer = io.BytesIO()

    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"gTTS synthesis failed: {exc}") from exc

    audio_bytes = buffer.getvalue()
    return base64.b64encode(audio_bytes).decode("utf-8")


def _generate_answer(question: str) -> tuple[str, Optional[str], Optional[str]]:
    import torch
    prompt = _build_prompt(question)
    inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=GENERATION_MAX_TOKENS,
            do_sample=False,
            repetition_penalty=1.5,
            stopping_criteria=STOPPING_CRITERIA,
        )

    raw_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    answer = _clean_response(raw_text)
    language = _detect_language(answer)
    audio_base64 = _synthesize_audio(answer, language)

    return answer, language, audio_base64


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="EduBot Backend", version="1.0.0")

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


@app.options("/api/chat")
async def options_chat():
    """Handle CORS preflight for /api/chat endpoint."""
    return {"message": "OK"}


@app.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "device": DEVICE.type if DEVICE else "cpu",
        "model": "bloomz-finetuned" if MODEL_AVAILABLE else "not-loaded",
        "model_available": str(MODEL_AVAILABLE),
    }


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Generate an answer and synthesized audio for the given question."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    if not MODEL_AVAILABLE:
        return ChatResponse(
            answer=(
                "The EduBot AI model is not loaded in this environment. "
                "To enable full functionality, place the 'bloomz-finetuned-model' "
                "directory in the project root. For now, please contact "
                "edubotofficial@gmail.com for campus queries."
            ),
            language="en",
            audio_base64=None,
        )

    loop = asyncio.get_event_loop()
    answer, language, audio_base64 = await loop.run_in_executor(
        None, _generate_answer, question
    )

    if not answer:
        raise HTTPException(
            status_code=500,
            detail="Model returned an empty response. Please try again.",
        )

    return ChatResponse(answer=answer, language=language, audio_base64=audio_base64)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Free GPU memory if applicable."""
    if DEVICE and DEVICE.type == "cuda":
        import torch
        torch.cuda.empty_cache()
