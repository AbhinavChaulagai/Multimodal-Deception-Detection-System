"""
Pre-downloads ML models during Render build phase.
Run once at build time so the live container never needs to fetch from HuggingFace.
"""
from pathlib import Path

CACHE = Path(__file__).parent / "model_cache"
(CACHE / "whisper").mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

print("Downloading faster-whisper tiny (int8)…")
from faster_whisper import WhisperModel
WhisperModel("tiny", device="cpu", compute_type="int8",
             download_root=str(CACHE / "whisper"))

print("Downloading MiniLM-L6-v2…")
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                    cache_folder=str(CACHE))

print("All models downloaded and cached.")
