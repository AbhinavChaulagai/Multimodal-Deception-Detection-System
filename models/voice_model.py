import torch
import torch.nn as nn

class VoiceModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Linear(64 * 32 * 32, 256)

    def forward(self, x):
        # x = spectrogram (B,1,H,W)

        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)