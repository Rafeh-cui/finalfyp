# EduBot - Smart Campus Assistant

A voice-enabled AI chatbot for COMSATS University Islamabad, Attock Campus. Students can ask campus-related questions and receive answers with synthesized audio.

## Architecture

- **Frontend**: Static HTML/CSS/JS pages served via `http-server` on port 5000
- **Backend**: Python FastAPI server on port 8000, providing `/api/chat` and `/health` endpoints

## Project Structure

```
├── frontend/           # Static HTML/JS/CSS pages
│   ├── index.html      # Entry redirect to firstscreen.html
│   ├── firstscreen.html  # App start/splash screen
│   ├── login.html      # Firebase login
│   ├── register.html   # Firebase registration
│   ├── welcomescreen.html  # Post-login welcome
│   ├── voicechat.html  # Main voice/text chat interface
│   ├── academics.html  # Academic info page
│   ├── faq.html        # FAQ page
│   ├── firebase.js     # Firebase auth/firestore config
│   └── registerHandler.js  # Registration logic
├── backend/
│   └── app/
│       └── main.py     # FastAPI app with BLOOMZ model integration
├── notebooks/          # Jupyter notebooks for AI/gTTS experiments
├── package.json        # npm scripts for dev server management
└── replit.md           # This file
```

## Workflows

- **Start application** (webview, port 5000): Serves the frontend static files
- **Backend API** (console, port 8000): FastAPI server with AI chat endpoint

## Key Dependencies

### Python (backend)
- `fastapi` + `uvicorn` - Web framework and ASGI server
- `transformers` - Hugging Face transformers (BLOOMZ model)
- `torch` - PyTorch (ML inference)
- `gTTS` - Google Text-to-Speech for audio responses
- `langdetect` - Language detection

### Node.js (frontend)
- `http-server` - Static file server
- `concurrently` - Run multiple processes

## AI Model

The backend requires a `bloomz-finetuned-model/` directory in the project root containing a fine-tuned BLOOMZ model. Without it, the backend runs in **demo mode** (returning a placeholder message). To enable full functionality, place the model files in that directory.

## Firebase

The frontend uses Firebase for authentication. The config is in `frontend/firebase.js` (project: `fyp056`).

## Environment Notes

- Backend runs in demo mode without the BLOOMZ model files
- Frontend Firebase auth will work with live internet connection
- `torch` and `transformers` are heavy dependencies; install only when model is available
