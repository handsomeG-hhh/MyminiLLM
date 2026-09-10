import torch
import torch.nn as nn

class VisualResampler(nn.Module):
    def __init__(
        self,
        vision_dim=1024,
        num_queries=32,
        num_heads=8
    ):
        super().__init__()
        self.query=nn.Parameter(
            torch.randn(
                1,
                num_queries,
                vision_dim
            )*0.02
        )
        self.attention=nn.MultiheadAttention(
            embed_dim=vision_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm=nn.LayerNorm(
            vision_dim
        )
    def forward(self,visual_features):
        batch_size=visual_features.size(0)
        query=self.query.expand(
            batch_size,
            -1,
            -1
        )
        output,_=self.attention(
            query,
            visual_features,
            visual_features
        )
        output=self.norm(
            output+query
        )
        return output