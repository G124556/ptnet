import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNLateralBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lateral = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.output = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, top_down: torch.Tensor = None) -> torch.Tensor:
        x = self.lateral(x)
        if top_down is not None:
            x = x + F.interpolate(top_down, size=x.shape[-2:], mode="nearest")
        return self.output(x)


class FPNHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, output_size: int):
        super().__init__()
        self.output_size = output_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return self.head(x)


class FPN(nn.Module):
    def __init__(
        self,
        in_channels: Tuple[int, ...],
        out_channels: int,
        num_levels: int,
        output_size: int,
        det_head_channels: int,
        spatial_tokens: int,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.out_channels = out_channels
        self.spatial_tokens = spatial_tokens
        self.grid_size = int(math.isqrt(spatial_tokens))

        self.laterals = nn.ModuleList([
            FPNLateralBlock(in_ch, out_channels)
            for in_ch in in_channels
        ])

        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.det_head = FPNHead(out_channels, det_head_channels, output_size)

    def _tokens_to_spatial(self, feat: torch.Tensor) -> torch.Tensor:
        B, N, D = feat.shape
        return feat.permute(0, 2, 1).reshape(B, D, self.grid_size, self.grid_size)

    def forward(
        self,
        O_det: torch.Tensor,
        level_features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_feats = [self._tokens_to_spatial(f) for f in level_features]

        top_down = None
        fpn_outs = []
        for i in range(self.num_levels - 1, -1, -1):
            feat = self.laterals[i](spatial_feats[i], top_down)
            fpn_outs.insert(0, feat)
            top_down = feat

        finest = fpn_outs[0]
        fused = finest
        for lvl in fpn_outs[1:]:
            fused = fused + F.interpolate(lvl, size=finest.shape[-2:], mode="nearest")

        fused = self.output_conv(fused)

        change_map = self.det_head(fused).squeeze(1)

        return change_map, fused
