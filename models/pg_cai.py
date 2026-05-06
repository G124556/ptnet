from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.prototype_bank import PrototypeBank


class PrototypeModulation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.W_Q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_K = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_V = nn.Linear(feature_dim, feature_dim, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        prototype: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = query.shape
        KN = prototype.size(0)

        Q = self.W_Q(query)
        K = self.W_K(prototype.unsqueeze(0).expand(B, -1, -1))
        V = self.W_V(prototype.unsqueeze(0).expand(B, -1, -1))

        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, KN, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, KN, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class CrossTemporalAttention(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.W_Q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_K = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_V = nn.Linear(feature_dim, feature_dim, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_feat: torch.Tensor,
        kv_feat: torch.Tensor,
        modulation: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = query_feat.shape

        Q = self.W_Q(query_feat)
        K = self.W_K(kv_feat * modulation)
        V = self.W_V(kv_feat * modulation)

        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class PGCAIBlock(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proto_modulation = PrototypeModulation(feature_dim, num_heads, dropout)

        self.change_query_mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

        self.cross_attn_1to2 = CrossTemporalAttention(feature_dim, num_heads, dropout)
        self.cross_attn_2to1 = CrossTemporalAttention(feature_dim, num_heads, dropout)

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm_u = nn.LayerNorm(feature_dim)

        self.ffn1 = nn.Sequential(
            nn.Linear(feature_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, feature_dim),
            nn.Dropout(dropout),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(feature_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, feature_dim),
            nn.Dropout(dropout),
        )

        self.norm_ffn1 = nn.LayerNorm(feature_dim)
        self.norm_ffn2 = nn.LayerNorm(feature_dim)

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        prototype: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        diff = torch.abs(f1 - f2)
        u = self.change_query_mlp(torch.cat([f1, f2, diff], dim=-1))
        u = self.norm_u(u)

        proto_flat = prototype.reshape(-1, prototype.size(-1))
        modulation = self.proto_modulation(u, proto_flat)

        g1 = self.norm1(f1 + self.cross_attn_1to2(f1, f2, modulation))
        g2 = self.norm2(f2 + self.cross_attn_2to1(f2, f1, modulation))

        g1 = self.norm_ffn1(g1 + self.ffn1(g1))
        g2 = self.norm_ffn2(g2 + self.ffn2(g2))

        return g1, g2


class PGCAILayer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            PGCAIBlock(feature_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        prototype: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            f1, f2 = block(f1, f2, prototype)
        return f1, f2


class PGCAI(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        num_levels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.layers = nn.ModuleList([
            PGCAILayer(feature_dim, num_heads, ffn_dim, num_layers, dropout)
            for _ in range(num_levels)
        ])

    def forward(
        self,
        features_1: dict,
        features_2: dict,
        prototype_bank: PrototypeBank,
        layer_keys: list,
    ) -> Tuple[dict, dict]:
        prototypes = prototype_bank()
        g1_dict, g2_dict = {}, {}

        for level_idx, key in enumerate(layer_keys):
            f1 = features_1[key]
            f2 = features_2[key]
            g1, g2 = self.layers[level_idx](f1, f2, prototypes)
            g1_dict[key] = g1
            g2_dict[key] = g2

        return g1_dict, g2_dict
