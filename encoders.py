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

