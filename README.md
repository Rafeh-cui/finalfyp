# EduBot Integration Project

## Developer
- Name: Muhammad Anas (integration)
- Role: Backend/Frontend integration and deployment support

This repository bundles the EduBot frontend, the fine-tuned BLOOMZ language model, and a FastAPI backend that exposes the model for text + TTS responses. The `voicechat.html` page consumes the backend to provide a campus assistant experience.

## Project Structure

- `backend/` – FastAPI service
  - `app/main.py` – loads the BLOOMZ model, exposes `/api/chat` for answers + speech, `/health` for status
  - `requirements.txt` – Python dependencies for the backend
- `frontend/` – static Tailwind HTML pages
  - `voicechat.html` – now wired to call the backend and play synthesized audio
  - other pages (`login.html`, `register.html`, etc.) remain Firebase-based but are not yet hooked to the backend
- `bloomz-finetuned-model/` – model weights, tokenizer files, and configs
- `notebooks/GTTS_Pipeline.ipynb` – original Colab notebook used to prototype inference + gTTS flow
- `package.json` – npm scripts to launch backend and serve the static frontend together

## Prerequisites

- Python 3.10 or later (for the FastAPI backend)
- Node.js 18+ / npm (for the convenience scripts)
- Model weights already present in `bloomz-finetuned-model/`

## Setup

1. **Python environment**

   ```bash
   cd backend
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   cd ..
   ```

2. **Install npm dependencies**

   ```bash
   npm install
   ```

## Run the system

After the above one-time setup, you can launch both servers together:

```bash
npm start
```

This command runs:

- FastAPI backend via Uvicorn on `http://localhost:8000`
- Static frontend server (http-server) on `http://localhost:3000`

Navigate to `http://localhost:3000/voicechat.html` to test the client. On send/voice input it will call the backend, receive the model’s answer, and play synthesized audio (returned as a base64 MP3).

## API Reference

- `GET /health` – simple health probe. Returns device (`cpu`/`cuda`) and model name.
- `POST /api/chat`
  - Request: `{ "question": "..." }`
  - Response: `{ "answer": "...", "language": "xx", "audio_base64": "..." }`

If gTTS fails to synthesize audio, the endpoint still returns the text answer.

## Notes & Next Steps

- The other frontend pages remain untouched; integrate them once their respective backend endpoints are ready.
- Model loading happens at service startup. Expect a short warm-up.
- When deploying, consider replacing `http-server` with your preferred static hosting stack and running the backend behind a production ASGI server (e.g., `gunicorn` + `uvicorn.workers.UvicornWorker`).


