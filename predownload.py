"""
Pre-downloads ML models during Render build phase.
Run once at build time so the live container never needs to fetch from the internet.
"""
import urllib.request
from pathlib import Path

import numpy as np

CACHE = Path(__file__).parent / "model_cache"
(CACHE / "whisper").mkdir(parents=True, exist_ok=True)

print("Downloading faster-whisper tiny (int8)…")
from faster_whisper import WhisperModel
WhisperModel("tiny", device="cpu", compute_type="int8",
             download_root=str(CACHE / "whisper"))

print("Downloading MiniLM-L6-v2…")
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                    cache_folder=str(CACHE))

print("Downloading MediaPipe FaceLandmarker model…")
landmarker_path = CACHE / "face_landmarker.task"
if not landmarker_path.exists():
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        landmarker_path,
    )

print("Verifying MediaPipe FaceLandmarker…")
import mediapipe as mp
base_options = mp.tasks.BaseOptions(model_asset_path=str(landmarker_path))
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1,
)
dummy = np.zeros((100, 100, 3), dtype=np.uint8)
mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy)
with mp.tasks.vision.FaceLandmarker.create_from_options(options) as lm:
    lm.detect(mp_img)

print("All models downloaded and cached.")
