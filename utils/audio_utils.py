import librosa
import torch

def audio_to_spectrogram(path):

    audio, sr = librosa.load(path)
    spec = librosa.feature.melspectrogram(audio, sr=sr)
    spec = torch.tensor(spec).unsqueeze(0)

    return spec