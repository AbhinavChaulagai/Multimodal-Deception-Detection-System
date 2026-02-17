import torch
import torch.nn as nn
from models.face_model import FaceModel
from models.voice_model import VoiceModel
from models.text_model import TextModel

class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.face_model = FaceModel()
        self.voice_model = VoiceModel()
        self.text_model = TextModel()

        self.classifier = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, video, audio, input_ids, attention_mask):

        face_feat = self.face_model(video)
        voice_feat = self.voice_model(audio)
        text_feat = self.text_model(input_ids, attention_mask)

        combined = torch.cat([face_feat, voice_feat, text_feat], dim=1)

        return self.classifier(combined)