import torch
from torch.utils.data import Dataset

class DeceptionDataset(Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        video = torch.randn(10, 3, 224, 224)
        audio = torch.randn(1, 128, 128)

        input_ids = torch.randint(0, 1000, (32,))
        attention_mask = torch.ones(32)

        label = torch.randint(0, 2, (1,)).item()

        return video, audio, input_ids, attention_mask, label