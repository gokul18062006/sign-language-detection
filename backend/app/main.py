from __future__ import annotations

import os
import re
import time
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from threading import RLock
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


APP_ROOT = Path(__file__).resolve().parents[1]

HAS_LEGACY_SOLUTIONS = hasattr(mp, "solutions")

if not HAS_LEGACY_SOLUTIONS:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision


def ensure_hand_landmarker_model() -> Path:
    model_dir = APP_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "hand_landmarker.task"
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    model_url = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )
    urllib.request.urlretrieve(model_url, model_path)
    return model_path


DEFAULT_FRONTEND_ORIGINS = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)

WORD_LIBRARY: dict[str, dict[str, str]] = {
    "hello": {
        "label": "Hello",
        "description": "A friendly open-hand greeting.",
        "gesture": "Five fingers",
    },
    "help": {
        "label": "Help",
        "description": "A clear signal when assistance is needed.",
        "gesture": "Four fingers",
    },
    "yes": {
        "label": "Yes",
        "description": "A quick positive confirmation.",
        "gesture": "Two fingers",
    },
    "no": {
        "label": "No",
        "description": "A simple negative response.",
        "gesture": "One finger",
    },
    "thank you": {
        "label": "Thank You",
        "description": "A polite expression of gratitude.",
        "gesture": "Phrase match",
    },
    "stop": {
        "label": "Stop",
        "description": "A firm signal to pause or halt.",
        "gesture": "Fist",
    },
}

FINGER_TO_WORD = {
    5: "hello",
    4: "help",
    2: "yes",
    1: "no",
    0: "stop",
}


def build_supported_word_list() -> list[str]:
    return list(WORD_LIBRARY.keys())


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def best_voice_match(text: str) -> Optional[str]:
    normalized = normalize_text(text)
    if not normalized:
        return None

    compact_normalized = normalized.replace(" ", "")
    words = build_supported_word_list()

    for word in sorted(words, key=len, reverse=True):
        if normalized == word:
            return word
        if normalized.startswith(word + " ") or normalized.endswith(" " + word):
            return word
        if word in normalized:
            return word
        if compact_normalized == word.replace(" ", ""):
            return word

    token_match = next((token for token in normalized.split() if token in WORD_LIBRARY), None)
    if token_match:
        return token_match

    compact_match = next(
        (
            word
            for word in words
            if word.replace(" ", "") in compact_normalized
            or compact_normalized in word.replace(" ", "")
        ),
        None,
    )
    if compact_match:
        return compact_match

    fuzzy = get_close_matches(normalized, words, n=1, cutoff=0.72)
    if fuzzy:
        return fuzzy[0]

    return None


def overlay_label(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    _, width = frame.shape[:2]
    box_width = min(width - 24, 520)
    box_height = 36 + 30 * len(lines)
    top_left = (12, 12)
    bottom_right = (12 + box_width, 12 + box_height)

    cv2.rectangle(frame, top_left, bottom_right, (8, 15, 28), thickness=-1)
    cv2.rectangle(frame, top_left, bottom_right, (58, 123, 213), thickness=2)

    y = 44
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (24, y + (index * 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (242, 247, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def create_placeholder_frame(title: str, subtitle: str) -> np.ndarray:
    frame = np.zeros((720, 960, 3), dtype=np.uint8)
    gradient = np.linspace(0, 1, frame.shape[1], dtype=np.float32)
    frame[:, :, 0] = (18 + gradient * 30).astype(np.uint8)
    frame[:, :, 1] = (24 + gradient * 18).astype(np.uint8)
    frame[:, :, 2] = (42 + gradient * 60).astype(np.uint8)

    center = (frame.shape[1] // 2, frame.shape[0] // 2 - 18)
    cv2.putText(
        frame,
        title,
        (center[0] - 260, center[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        (246, 248, 252),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        subtitle,
        (center[0] - 300, center[1] + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (191, 219, 254),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Connect a webcam and start sign detection to stream live hand tracking.",
        (center[0] - 330, center[1] + 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (203, 213, 225),
        2,
        cv2.LINE_AA,
    )
    return frame


@dataclass
class AppState:
    lock: RLock = field(default_factory=RLock, init=False, repr=False)
    sign_active: bool = False
    voice_active: bool = False
    detected_word: Optional[str] = None
    gesture_label: str = "Idle"
    voice_transcript: Optional[str] = None
    matched_word: Optional[str] = None
    last_event: str = "Ready"
    last_detection_time: Optional[float] = None
    logs: list[str] = field(default_factory=list)

    def add_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.logs = [f"[{timestamp}] {message}", *self.logs[:19]]
        self.last_event = message

    def set_sign_active(self, active: bool) -> None:
        self.sign_active = active
        self.add_log(f"Sign detection {'started' if active else 'stopped'}")
        if not active:
            self.detected_word = None
            self.gesture_label = "Detection paused"

    def set_voice_active(self, active: bool) -> None:
        self.voice_active = active
        self.add_log(f"Voice listening {'started' if active else 'stopped'}")

    def update_detection(self, word: Optional[str], gesture_label: str) -> None:
        self.gesture_label = gesture_label
        self.last_detection_time = time.time()
        if word != self.detected_word:
            self.detected_word = word
            if word:
                self.add_log(f"Detected sign translated to '{word}'")

    def update_voice_result(self, transcript: str, matched_word: Optional[str]) -> None:
        self.voice_transcript = transcript
        self.matched_word = matched_word
        if matched_word:
            self.add_log(f"Voice input matched '{matched_word}'")
        else:
            self.add_log("Voice input did not match a supported word")

    def snapshot(self, camera_ready: bool) -> dict[str, object]:
        with self.lock:
            detected_definition = WORD_LIBRARY.get(self.detected_word or "")
            matched_definition = WORD_LIBRARY.get(self.matched_word or "")
            return {
                "sign_active": self.sign_active,
                "voice_active": self.voice_active,
                "camera_ready": camera_ready,
                "detected_word": self.detected_word,
                "detected_label": detected_definition["label"] if detected_definition else None,
                "gesture_label": self.gesture_label,
                "voice_transcript": self.voice_transcript,
                "matched_word": self.matched_word,
                "matched_label": matched_definition["label"] if matched_definition else None,
                "last_event": self.last_event,
                "last_detection_time": self.last_detection_time,
                "logs": self.logs,
                "supported_words": build_supported_word_list(),
            }


class VoiceInput(BaseModel):
    transcript: str


class CameraService:
    def __init__(self) -> None:
        backend = cv2.CAP_DSHOW if os.name == "nt" and hasattr(cv2, "CAP_DSHOW") else 0
        self.capture = cv2.VideoCapture(0, backend) if backend else cv2.VideoCapture(0)
        self.lock = RLock()

    @property
    def ready(self) -> bool:
        return bool(self.capture and self.capture.isOpened())

    def read(self) -> Optional[np.ndarray]:
        if not self.ready:
            return None
        with self.lock:
            success, frame = self.capture.read()
        if not success:
            return None
        return frame

    def close(self) -> None:
        with self.lock:
            if self.capture and self.capture.isOpened():
                self.capture.release()


class GestureRecognizer:
    def __init__(self) -> None:
        self.use_legacy_solutions = HAS_LEGACY_SOLUTIONS
        self.hands = None
        self.landmarker = None
        self.drawer = None
        self.style = None

        if self.use_legacy_solutions:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.55,
            )
            self.drawer = mp.solutions.drawing_utils
            self.style = mp.solutions.drawing_styles
        else:
            model_path = ensure_hand_landmarker_model()
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.55,
                min_tracking_confidence=0.55,
            )
            self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

        self.handedness_label = "Unknown"

    def close(self) -> None:
        if self.hands:
            self.hands.close()
        if self.landmarker:
            self.landmarker.close()

    @staticmethod
    def _distance(a, b) -> float:
        return float(np.hypot(a.x - b.x, a.y - b.y))

    def _thumb_open(self, landmarks) -> bool:
        # Detect thumb extension by comparing thumb reach against palm scale.
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]

        palm_width = self._distance(index_mcp, pinky_mcp)
        tip_to_index = self._distance(thumb_tip, index_mcp)
        ip_to_index = self._distance(thumb_ip, index_mcp)
        tip_to_wrist = self._distance(thumb_tip, wrist)
        ip_to_wrist = self._distance(thumb_ip, wrist)

        return (
            tip_to_index > (ip_to_index * 1.15)
            and tip_to_wrist > (ip_to_wrist * 1.05)
            and tip_to_index > (palm_width * 0.36)
        )

    def _finger_extended(self, landmarks, tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
        wrist = landmarks[0]
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        mcp = landmarks[mcp_idx]

        tip_above_joint = tip.y < (pip.y - 0.015)
        tip_farther_than_pip = self._distance(tip, wrist) > (self._distance(pip, wrist) * 1.08)
        tip_farther_than_mcp = self._distance(tip, wrist) > (self._distance(mcp, wrist) * 1.03)

        return tip_above_joint and tip_farther_than_pip and tip_farther_than_mcp

    def _count_fingers(self, landmarks) -> int:
        finger_states = [
            self._thumb_open(landmarks),
            self._finger_extended(landmarks, 8, 6, 5),
            self._finger_extended(landmarks, 12, 10, 9),
            self._finger_extended(landmarks, 16, 14, 13),
            self._finger_extended(landmarks, 20, 18, 17),
        ]
        return sum(1 for state in finger_states if state)

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, Optional[str], str]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        candidate_word: Optional[str] = None
        gesture_label = "No hand detected"

        if self.use_legacy_solutions and self.hands:
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = "Unknown"
                if results.multi_handedness:
                    handedness = results.multi_handedness[0].classification[0].label
                self.handedness_label = handedness

                finger_count = self._count_fingers(hand_landmarks.landmark)
                candidate_word = FINGER_TO_WORD.get(finger_count)
                if candidate_word:
                    gesture_label = f"Fingers: {finger_count} -> {WORD_LIBRARY[candidate_word]['label']}"
                else:
                    gesture_label = f"Fingers: {finger_count} -> No mapped word"

                self.drawer.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    self.style.get_default_hand_landmarks_style(),
                    self.style.get_default_hand_connections_style(),
                )
        elif self.landmarker:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.landmarker.detect(mp_image)
            hand_landmarks = results.hand_landmarks[0] if results.hand_landmarks else None
            if hand_landmarks:
                handedness = "Unknown"
                if results.handedness and len(results.handedness[0]) > 0:
                    category = results.handedness[0][0]
                    handedness = (
                        getattr(category, "category_name", None)
                        or getattr(category, "display_name", None)
                        or "Unknown"
                    )
                self.handedness_label = handedness

                finger_count = self._count_fingers(hand_landmarks)
                candidate_word = FINGER_TO_WORD.get(finger_count)
                if candidate_word:
                    gesture_label = f"Fingers: {finger_count} -> {WORD_LIBRARY[candidate_word]['label']}"
                else:
                    gesture_label = f"Fingers: {finger_count} -> No mapped word"

                for landmark in hand_landmarks:
                    cx = int(landmark.x * frame.shape[1])
                    cy = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 4, (91, 231, 221), -1)

        overlay_label(
            frame,
            [
                "Sign Detection",
                gesture_label,
                f"Handedness: {self.handedness_label}",
            ],
        )
        return frame, candidate_word, gesture_label


class GestureSmoother:
    def __init__(
        self,
        stable_frames: int = 5,
        switch_frames: int = 7,
        window_size: int = 12,
        empty_frames: int = 10,
    ) -> None:
        self.stable_frames = stable_frames
        self.switch_frames = switch_frames
        self.window_size = window_size
        self.empty_frames = empty_frames
        self.last_candidate: Optional[str] = None
        self.candidate_count = 0
        self.stable_word: Optional[str] = None
        self.empty_count = 0
        self.history: deque[Optional[str]] = deque(maxlen=window_size)

    def reset(self) -> None:
        self.last_candidate = None
        self.candidate_count = 0
        self.stable_word = None
        self.empty_count = 0
        self.history.clear()

    def update(self, candidate_word: Optional[str]) -> Optional[str]:
        self.history.append(candidate_word)

        if candidate_word == self.last_candidate:
            self.candidate_count += 1
        else:
            self.last_candidate = candidate_word
            self.candidate_count = 1

        if candidate_word is None:
            self.empty_count += 1
        else:
            self.empty_count = 0

        non_empty = [item for item in self.history if item is not None]
        if non_empty:
            most_common_word, most_common_count = Counter(non_empty).most_common(1)[0]
            confidence = most_common_count / max(1, len(self.history))

            if self.stable_word is None:
                if (
                    most_common_count >= self.stable_frames
                    and confidence >= 0.55
                ):
                    self.stable_word = most_common_word
            elif most_common_word != self.stable_word:
                if (
                    most_common_count >= self.switch_frames
                    and confidence >= 0.65
                ):
                    self.stable_word = most_common_word

        if candidate_word is None and self.empty_count >= self.empty_frames:
            self.stable_word = None

        return self.stable_word


class SignLanguageService:
    def __init__(self) -> None:
        self.state = AppState()
        self.camera = CameraService()
        self.recognizer = GestureRecognizer()
        self.smoother = GestureSmoother()

    def close(self) -> None:
        self.camera.close()
        self.recognizer.close()

    def encode_frame(self, frame: np.ndarray) -> bytes:
        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        if not success:
            raise RuntimeError("Unable to encode camera frame")
        return buffer.tobytes()

    def iter_frames(self):
        while True:
            frame = self.camera.read()
            if frame is None:
                placeholder = create_placeholder_frame(
                    "Camera unavailable",
                    "Allow camera access or connect a webcam to begin.",
                )
                with self.state.lock:
                    if self.state.sign_active:
                        self.state.update_detection(None, "Camera unavailable")
                        self.smoother.reset()
                frame = placeholder
            else:
                with self.state.lock:
                    if self.state.sign_active:
                        frame, candidate_word, gesture_label = self.recognizer.process(frame)
                        stable_word = self.smoother.update(candidate_word)
                        self.state.update_detection(stable_word, gesture_label)
                    else:
                        self.state.gesture_label = "Detection paused"
                        self.smoother.reset()
                        overlay_label(frame, ["Sign Detection", "Detection paused", "Click Start Detection"])

            jpeg_bytes = self.encode_frame(frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
            time.sleep(0.04)

    def start_sign(self) -> dict[str, object]:
        with self.state.lock:
            self.state.set_sign_active(True)
            self.smoother.reset()
            return self.state.snapshot(self.camera.ready)

    def stop_sign(self) -> dict[str, object]:
        with self.state.lock:
            self.state.set_sign_active(False)
            self.smoother.reset()
            return self.state.snapshot(self.camera.ready)

    def start_voice(self) -> dict[str, object]:
        with self.state.lock:
            self.state.set_voice_active(True)
            return self.state.snapshot(self.camera.ready)

    def stop_voice(self) -> dict[str, object]:
        with self.state.lock:
            self.state.set_voice_active(False)
            return self.state.snapshot(self.camera.ready)

    def translate_voice(self, transcript: str) -> dict[str, object]:
        normalized = transcript.strip()
        if not normalized:
            raise HTTPException(status_code=400, detail="Transcript cannot be empty")

        matched_word = best_voice_match(normalized)
        with self.state.lock:
            self.state.update_voice_result(normalized, matched_word)

        matched_definition = WORD_LIBRARY.get(matched_word or "")
        return {
            "transcript": normalized,
            "normalized": normalize_text(normalized),
            "matched_word": matched_word,
            "matched_label": matched_definition["label"] if matched_definition else None,
            "description": matched_definition["description"] if matched_definition else None,
            "supported_words": build_supported_word_list(),
        }


service = SignLanguageService()

app = FastAPI(title="Two-Way Sign Language Bridge", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in DEFAULT_FRONTEND_ORIGINS.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
def shutdown_services() -> None:
    service.close()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict[str, object]:
    return service.state.snapshot(service.camera.ready)


@app.get("/api/start-sign")
def start_sign() -> dict[str, object]:
    return service.start_sign()


@app.get("/api/stop-sign")
def stop_sign() -> dict[str, object]:
    return service.stop_sign()


@app.get("/api/start-voice")
def start_voice() -> dict[str, object]:
    return service.start_voice()


@app.get("/api/stop-voice")
def stop_voice() -> dict[str, object]:
    return service.stop_voice()


@app.post("/api/voice-text")
def voice_text(payload: VoiceInput) -> dict[str, object]:
    return service.translate_voice(payload.transcript)


@app.get("/api/video-feed")
def video_feed() -> StreamingResponse:
    return StreamingResponse(
        service.iter_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Two-Way Sign Language Bridge",
        "frontend": "Run the React app in /frontend and connect it to this API.",
    }
