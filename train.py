import torch
from torch.utils.data import DataLoader
from dataset import DeceptionDataset
from models.multimodal_model import MultimodalModel
import config

dataset = DeceptionDataset()
loader = DataLoader(dataset, batch_size=config.BATCH_SIZE)

model = MultimodalModel().to(config.DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

for epoch in range(config.EPOCHS):

    for video, audio, ids, mask, label in loader:

        video = video.to(config.DEVICE)
        audio = audio.to(config.DEVICE)
        ids = ids.to(config.DEVICE)
        mask = mask.to(config.DEVICE)
        label = label.to(config.DEVICE)

        output = model(video, audio, ids, mask)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss: {loss.item()}")