# EduBot — COMSATS University Islamabad, Attock Campus

AI-powered voice chatbot for COMSATS University Islamabad, Attock Campus students. Built as a Final Year Project (FYP).

## Architecture

**Single-server setup**: FastAPI (Python) runs on port 5000 and serves BOTH the frontend static files AND the `/api/chat` AI endpoint.

- `backend/app/main.py` — FastAPI app (Gemini AI + gTTS + static file serving)
- `frontend/` — Static HTML/CSS/JS pages (served by FastAPI's StaticFiles mount)

## Pages

| Page | File | Purpose |
|------|------|---------|
| First Screen | `frontend/firstscreen.html` | Landing/splash screen |
| Login | `frontend/login.html` | Firebase Auth login |
| Register | `frontend/register.html` | Firebase Auth registration |
| Welcome | `frontend/welcomescreen.html` | Home with quick access cards |
| Voice Chat | `frontend/voicechat.html` | AI chat (text + voice input) |
| Academics | `frontend/academics.html` | 6 category cards with modals |
| FAQ | `frontend/faq.html` | Expandable accordion FAQs |

## Tech Stack

- **AI**: Google Gemini API (`gemini-2.0-flash-lite` → `gemini-2.0-flash` → `gemini-2.5-flash` fallback chain)
- **Backend**: FastAPI + Python 3.11 + gTTS (text-to-speech) + langdetect
- **Frontend**: HTML + TailwindCSS CDN + vanilla JS + Web Speech API
- **Auth**: Firebase Authentication (project: fyp056)
- **Hosting**: Replit (port 5000)

## Workflow

- **Start application**: `python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 5000 --reload`

## Environment Variables / Secrets

- `GEMINI_API_KEY` — Google Gemini API key (set in Replit Secrets)

## Key Design Decisions

- FastAPI serves both API and frontend on port 5000 to avoid cross-origin URL issues in Replit's proxied iframe environment
- Frontend uses relative URL `/api/chat` for all AI requests
- Gemini model fallback chain handles quota exhaustion gracefully
- 20-second timeout on all AI requests to prevent hanging

## Navigation Flow

firstscreen → login (or register) → welcomescreen → voicechat / academics / faq
