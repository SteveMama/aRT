import torch
from encoders import *
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer
import torchvision.models as models
import torchaudio


class ModalityEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.video_encoder = models.video.r3d_18(pretrained=True)
        self.video_encoder.fc = nn.Linear(512, config.hidden_size)

        self.audio_encoder = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft = 400,
                win_length=400,
                hop_length=160,
                n_mels=80
            ),
            nn.Conv2d(1, 32, kernel_size=3, stride= 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 20 *20, config.hidden_size)
        )