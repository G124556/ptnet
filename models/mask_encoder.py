from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        pool_size: Tuple[int, int],
        hidden_dim: int,
        output_dim: int,
        num_mlp_layers: int = 2,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.num_tokens = pool_size[0] * pool_size[1]

        flat_dim = input_channels * pool_size[0] * pool_size[1]

        layers = []
        in_dim = input_channels
        for i in range(num_mlp_layers):
            out_dim = hidden_dim if i < num_mlp_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_mlp_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, fpn_feat: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(fpn_feat, self.pool_size)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.mlp(x)
        x = self.norm(x)
        return x
