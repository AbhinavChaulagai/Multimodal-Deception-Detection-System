"""
Multimodal deception detection model.

Three branches:
  - Video:  MediaPipe FaceMesh landmarks → LSTM → 64-dim feature
  - Audio:  librosa MFCCs → Conv1D + LSTM → 64-dim feature
  - Text:   MiniLM-L6-v2 embedding → MLP → 64-dim feature

Fusion: concatenate (192-dim) → MLP → binary logit
"""

import torch
import torch.nn as nn

VIDEO_LANDMARK_DIM = 478 * 3   # 1434
N_MFCC             = 40
TEXT_EMBED_DIM     = 384


class VideoEncoder(nn.Module):
    def __init__(self, input_dim=VIDEO_LANDMARK_DIM, hidden_dim=256, out_dim=64, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.fc   = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, out_dim),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out.mean(dim=1))


class AudioEncoder(nn.Module):
    def __init__(self, n_mfcc=N_MFCC, hidden_dim=128, out_dim=64, dropout=0.4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mfcc, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),     nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True, num_layers=1)
        self.fc   = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out.mean(dim=1))


class TextEncoder(nn.Module):
    def __init__(self, input_dim=TEXT_EMBED_DIM, out_dim=64, dropout=0.4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.fc(x)


class DeceptionModel(nn.Module):
    def __init__(self, video_input_dim=VIDEO_LANDMARK_DIM, n_mfcc=N_MFCC,
                 text_input_dim=TEXT_EMBED_DIM, hidden_dim=64, dropout=0.4):
        super().__init__()
        self.video_encoder = VideoEncoder(video_input_dim, 256, hidden_dim, dropout)
        self.audio_encoder = AudioEncoder(n_mfcc, 128, hidden_dim, dropout)
        self.text_encoder  = TextEncoder(text_input_dim, hidden_dim, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),             nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, video, audio, text):
        v = self.video_encoder(video)
        a = self.audio_encoder(audio)
        t = self.text_encoder(text)
        return self.fusion(torch.cat([v, a, t], dim=1))

    @torch.no_grad()
    def predict_proba(self, video, audio, text):
        return torch.sigmoid(self.forward(video, audio, text)).squeeze(-1)


MODEL_CONFIG = {
    "video_input_dim": VIDEO_LANDMARK_DIM,
    "n_mfcc":          N_MFCC,
    "text_input_dim":  TEXT_EMBED_DIM,
    "hidden_dim":      64,
    "dropout":         0.4,
}
