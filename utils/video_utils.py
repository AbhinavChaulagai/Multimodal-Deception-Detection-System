import cv2
import torch

def load_video_frames(path, num_frames=10):
    cap = cv2.VideoCapture(path)

    frames = []

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = torch.tensor(frame).permute(2,0,1) / 255
        frames.append(frame)

    cap.release()

    return torch.stack(frames)