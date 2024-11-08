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

        self.text_encoder = AutoModel.from_pretrained('bert-base-uncase')
        self.text_projection = nn.Linear(768, config.hidden_size)

    def forward(self, video, audio, text, attention_mask = None):

        video_feat = self.video_encoder(video)
        audio_feat = self.audio_encoder(audio)

        text_feat = self.text_encoder(
            input_ids = text,
            attention_mask = attention_mask
        ).last_hidden_state[:, 0, :]
        text_feat = self.text_projection(text_feat)

        return video_feat, audio_feat, text_feat


class CrossModalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim= config.hidden_size,
            num_heads = 8,
            dropout=0.1
        )

    def forward(self, query, key, value):
        return self.attention(query, key, value)[0]