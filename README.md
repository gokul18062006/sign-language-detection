# Two-Way Sign Language Bridge

This project is a working full-stack starter for bidirectional communication between sign gestures and spoken words. The frontend is React + TypeScript, and the backend is FastAPI with MediaPipe and OpenCV for webcam-based gesture recognition.

## What it does

### Sign to Voice
- Streams live webcam video from the FastAPI backend
- Detects simple hand-count gestures with MediaPipe
- Maps supported gestures to words
- Uses browser speech synthesis to speak the detected word

### Voice to Sign
- Uses the browser Speech Recognition API for microphone input
- Sends recognized text to the FastAPI backend for matching
- Shows a sign card for the matched word
- Keeps a running activity log and live status panel

### Supported mappings
- 5 fingers -> hello
- 4 fingers -> help
- 2 fingers -> yes
- 1 finger -> no
- Fist -> stop
- Spoken phrase -> thank you

## Project structure

```
.
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── styles.css
│   │   └── vite-env.d.ts
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── vite.config.ts
└── README.md
```

## Requirements

- Python 3.10+
- Node.js 18+
- Webcam for sign detection
- Microphone for browser speech recognition
- Chromium-based browser for the best voice input support

## Backend setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL in your browser, usually `http://localhost:5173`.

## Usage

1. Start the backend and the frontend.
2. Open the app in your browser and allow camera access.
3. Click Start Detection in the sign panel and show one of the supported gestures.
4. Click Start Listening in the voice panel and speak a supported phrase.
5. Use the stop buttons to pause either mode.

## API endpoints

- `GET /api/health` - Health check
- `GET /api/status` - Current backend state
- `GET /api/start-sign` - Enable sign detection
- `GET /api/stop-sign` - Disable sign detection
- `GET /api/video-feed` - MJPEG webcam stream
- `GET /api/start-voice` - Mark voice mode active
- `GET /api/stop-voice` - Mark voice mode inactive
- `POST /api/voice-text` - Match recognized transcript to a supported word

## Notes

- The browser handles voice recognition and speech output, which avoids native microphone and TTS setup issues on Windows.
- If no webcam is available, the backend serves a clear placeholder stream instead of failing.
- The frontend can be pointed at a different backend URL by setting `VITE_API_BASE_URL`.
