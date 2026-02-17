import torch
import torch.nn as nn
import torchvision.models as models

class FaceModel(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            batch_first=True
        )

    def forward(self, x):
        # x = (B, T, C, H, W)

        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)

        features = self.cnn(x)
        features = features.view(B, T, -1)

        _, (hidden, _) = self.lstm(features)

        return hidden[-1]