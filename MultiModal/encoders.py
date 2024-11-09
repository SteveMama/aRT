import torch
import torch.nn as nn

class VideoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        self.fc = nn.Linear(input_dim, hidden_dim)

class AudioEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3),
            nn.ReLU()
        )
        self.fc = nn.Linear(256, hidden_dim)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 300)
        self.lstm = nn.LSTM(300, hidden_dim, batch_first=True)