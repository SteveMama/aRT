from MultiModal import *

class MultimodalFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttention(config) for _ in range(3)
        ])

        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size * 2),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )

        self.residual_projection = nn.Linear(config.hidden_size * 3, config.hidden_size)



    def forward(self, video_feat, audio_feat, text_feat):
        video_attn = self.cross_attention_layers[0](
            video_feat,
            torch.stack([audio_feat, text_feat]),
            torch.stack([audio_feat, text_feat])
        )

        audio_attn = self.cross_attention_layers[1](
            audio_feat,
            torch.stack([video_feat, text_feat]),
            torch.stack([video_feat, text_feat])
        )

        text_attn = self.cross_attention_layers[2](
            text_feat,
            torch.stack([video_feat, audio_feat]),
            torch.stack([video_feat, audio_feat])
        )

        combined = torch.cat(
            [video_attn, audio_attn, text_attn],
            dim =-1
        )

        residual = self.residual_projection(combined)
        fused = self.fusion_layer(combined)

        return fused

class InterruptionPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.modality_encoder = ModalityEncoder(config)
        self.fusion_network = MultimodalFusion(config)

        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size // 2),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_size // 4),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )

    @autocast
    def forward(self, video, audio, text, attention_mask = None):

        video_feat, audio_feat, text_feat = self.modality_encoder(
            video, audio, text, attention_mask
        )

        fused_feat = self.fusion_network(
            video_feat,
            audio_feat,
            text_feat
        )

        interruption_rate = self.regression_head(fused_feat)

        return interruption_rate