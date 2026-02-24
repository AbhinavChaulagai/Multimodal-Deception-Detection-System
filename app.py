"""
FastAPI backend — Deception Detection API

Endpoints:
  GET  /health   → liveness check
  POST /predict  → accepts a video file, returns deception score

Environment variables (set in Render dashboard):
  SUPABASE_URL   — your Supabase project URL
  SUPABASE_KEY   — your Supabase service-role key
"""

import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import librosa
import numpy as np
import torch
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from model import DeceptionModel, MODEL_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (must match training/preprocess.py)
# ---------------------------------------------------------------------------
NUM_VIDEO_FRAMES = 30
AUDIO_DURATION   = 10
SAMPLE_RATE      = 22050
N_MFCC           = 40
HOP_LENGTH       = 512
MAX_AUDIO_FRAMES = int(AUDIO_DURATION * SAMPLE_RATE / HOP_LENGTH)  # ≈430
MAX_UPLOAD_BYTES = 60 * 1024 * 1024  # 60 MB

MODEL_WEIGHTS = Path(__file__).parent / "model_weights.pth"

# ---------------------------------------------------------------------------
# Global model registry (loaded once at startup)
# ---------------------------------------------------------------------------
_M: dict = {}


def _load_all_models():
    log.info("Loading Whisper tiny (faster-whisper, int8 CPU)…")
    _M["whisper"] = WhisperModel("tiny", device="cpu", compute_type="int8")

    log.info("Loading MiniLM sentence encoder…")
    _M["minilm"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    log.info("Loading PyTorch deception model…")
    if not MODEL_WEIGHTS.exists():
        raise RuntimeError(f"Model weights not found at {MODEL_WEIGHTS}. Train first.")
    ckpt  = torch.load(MODEL_WEIGHTS, map_location="cpu")
    cfg   = ckpt.get("model_config", MODEL_CONFIG)
    model = DeceptionModel(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _M["model"] = model
    cv_auc = ckpt.get("cv_auc")
    log.info(f"  CV AUC from training: {cv_auc:.3f}" if cv_auc is not None else "  CV AUC from training: N/A")

    # Supabase (optional)
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_KEY", "")
    if supabase_url and supabase_key:
        try:
            from supabase import create_client
            _M["supabase"] = create_client(supabase_url, supabase_key)
            log.info("Supabase client initialised.")
        except Exception as exc:
            log.warning(f"Supabase init failed: {exc}")
            _M["supabase"] = None
    else:
        log.warning("SUPABASE_URL / SUPABASE_KEY not set — cloud storage disabled.")
        _M["supabase"] = None

    log.info("All models ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all_models()
    yield
    _M.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Deception Detection API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten to your Netlify domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _extract_landmarks(video_path: str) -> np.ndarray:
    """Video branch disabled — return zeros to match training data."""
    return np.zeros((NUM_VIDEO_FRAMES, 478 * 3), dtype=np.float32)


def _extract_mfcc(video_path: str) -> np.ndarray:
    try:
        y, _ = librosa.load(video_path, sr=SAMPLE_RATE, mono=True, duration=AUDIO_DURATION)
        mfcc  = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH).T
        if len(mfcc) < MAX_AUDIO_FRAMES:
            mfcc = np.vstack([mfcc, np.zeros((MAX_AUDIO_FRAMES - len(mfcc), N_MFCC))])
        else:
            mfcc = mfcc[:MAX_AUDIO_FRAMES]
        return mfcc.astype(np.float32)
    except Exception as exc:
        log.error(f"MFCC extraction failed: {exc}")
        return np.zeros((MAX_AUDIO_FRAMES, N_MFCC), dtype=np.float32)


def _transcribe(video_path: str) -> str:
    try:
        segments, _ = _M["whisper"].transcribe(video_path)
        return " ".join(s.text for s in segments).strip()
    except Exception as exc:
        log.error(f"Transcription failed: {exc}")
        return ""


def _embed_text(text: str) -> np.ndarray:
    if not text:
        text = "no speech detected"
    return _M["minilm"].encode(text, normalize_embeddings=True).astype(np.float32)


def _store_result(file_bytes: bytes, filename: str, score: float, transcript: str):
    client = _M.get("supabase")
    if client is None:
        return
    try:
        client.storage.from_("recordings").upload(
            filename, file_bytes, {"content-type": "video/webm"}
        )
        client.table("predictions").insert({
            "id":         str(uuid.uuid4()),
            "filename":   filename,
            "score":      score,
            "label":      "Deceptive" if score >= 0.5 else "Truthful",
            "transcript": transcript,
        }).execute()
        log.info(f"Stored result → Supabase: {filename}")
    except Exception as exc:
        log.error(f"Supabase storage error: {exc}")


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    score:      float   # 0.0 (very truthful) → 1.0 (very deceptive)
    label:      str     # "Truthful" or "Deceptive"
    confidence: str     # "High" | "Low"
    transcript: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(_M)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Expected a video/* file.")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 60 MB limit.")

    # Determine suffix from content-type
    suffix = ".webm" if "webm" in file.content_type else ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        log.info(f"Extracting features from {len(file_bytes)//1024} KB video…")

        video_feat = _extract_landmarks(tmp_path)   # (30, 1434)
        audio_feat = _extract_mfcc(tmp_path)         # (T_a, 40)
        transcript = _transcribe(tmp_path)
        text_emb   = _embed_text(transcript)         # (384,)

        log.info(f"Transcript: {transcript[:120]!r}")

        # Inference
        model = _M["model"]
        with torch.no_grad():
            v      = torch.from_numpy(video_feat).unsqueeze(0)  # (1, 30, 1434)
            a      = torch.from_numpy(audio_feat).unsqueeze(0)  # (1, T_a, 40)
            t      = torch.from_numpy(text_emb).unsqueeze(0)    # (1, 384)
            logit  = model(v, a, t)
            score  = float(torch.sigmoid(logit).item())

        label      = "Deceptive" if score >= 0.5 else "Truthful"
        confidence = "High" if abs(score - 0.5) > 0.2 else "Low"
        log.info(f"Score: {score:.3f}  → {label} ({confidence} confidence)")

        # Cloud storage (non-blocking — errors are logged, not raised)
        filename = f"{uuid.uuid4()}{suffix}"
        _store_result(file_bytes, filename, score, transcript)

        return PredictionResponse(
            score=round(score, 4),
            label=label,
            confidence=confidence,
            transcript=transcript,
        )

    finally:
        Path(tmp_path).unlink(missing_ok=True)
