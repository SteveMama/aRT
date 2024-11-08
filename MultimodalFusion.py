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

        fused = self.fusion_layer(combined)

        return fused