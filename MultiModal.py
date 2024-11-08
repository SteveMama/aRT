import torch
from encoders import *
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer
import torchvision.models as models
import torchaudio
from torch.cpu.amp import autocast
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len= 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0 :: 2] = torch.sin(position * div_term)
        pe[:,  1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ModalityEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.video_encoder = models.efficientnet_b0(pretrained=True)
        self.video_encoder.classifier = nn.Sequential(
            nn.Dropout(p = 0.2, inplace = True),
            nn.Linear(1280, config.hidden_size)
        )


        self.audio_encoder = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft = 400,
                win_length=400,
                hop_length=160,
                n_mels=80,
                normalized=True
            ),
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride =2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(64 * 16, config.hidden_size),
            nn.Dropout(0.2)
        )

        self.text_encoder = AutoModel.from_pretrained('distilbert-base-uncased')
        self.text_projection = nn.Sequential(
            nn.Linear(768, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(0.1)
        )

        self.pos_encoder = PositionalEncoding(config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)


    @autocast
    def forward(self, video, audio, text, attention_mask = None):

        batch_size = video.size(0)
        video_feat = self.video_encoder(video.view(-1,3, 224, 224))
        video_feat = video_feat.view(batch_size, -1, self.config.hidden_size)
        video_feat = self.pos_encoder(video_feat)
        video_feat = self.layer_norm(video_feat)


        audio_feat = self.audio_encoder(audio)
        audio_feat = self.layer_norm(audio_feat)

        text_feat = self.text_encoder(
            input_ids = text,
            attention_mask = attention_mask
        ).last_hidden_state[:, 0, :]
        text_feat = self.text_projection(text_feat)
        text_feat = self.layer_norm(text_feat)

        return video_feat, audio_feat, text_feat


class CrossModalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim= config.hidden_size,
            num_heads = 8,
            dropout=0.1,
            batch_first= True
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, key_padding_mask= None):
        attn_output, _ = self.attention(
            query, key, value,
            key_padding_mask = key_padding_mask,
            need_weights = False
        )

        return self.layer_norm(query + self.dropout(attn_output))