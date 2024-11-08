from MultiModal import *

class MultimodalFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttention(config) for _ in range(3)
        ])


        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)

        )
