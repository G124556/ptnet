from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadWiseGate(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.gate_fc = nn.Linear(self.head_dim, 1, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, N, D = feat.shape
        feat_heads = feat.reshape(B, N, self.num_heads, self.head_dim)

        pooled = feat_heads.mean(dim=1)
        gate_logits = self.gate_fc(pooled).squeeze(-1)
        gates = torch.sigmoid(gate_logits)

        gates = self.dropout(gates)
        gates = gates.unsqueeze(1).unsqueeze(-1)

        gated = feat_heads * gates
        return gated.reshape(B, N, D)


class TAMGFusion(nn.Module):
    def __init__(self, feature_dim: int, num_levels: int, num_temporal: int):
        super().__init__()
        total = num_levels * num_temporal
        self.scale_logits = nn.Parameter(torch.zeros(total))

    def forward(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        weights = F.softmax(self.scale_logits, dim=0)
        out = sum(w * f for w, f in zip(weights, feature_list))
        return out


class TAMG(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        num_levels: int,
        num_temporal: int,
        gate_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.num_temporal = num_temporal

        self.det_gates = nn.ModuleList([
            nn.ModuleList([
                HeadWiseGate(feature_dim, num_heads, gate_dropout)
                for _ in range(num_temporal)
            ])
            for _ in range(num_levels)
        ])

        self.cap_gates = nn.ModuleList([
            nn.ModuleList([
                HeadWiseGate(feature_dim, num_heads, gate_dropout)
                for _ in range(num_temporal)
            ])
            for _ in range(num_levels)
        ])

        self.det_fusion = TAMGFusion(feature_dim, num_levels, num_temporal)
        self.cap_fusion = TAMGFusion(feature_dim, num_levels, num_temporal)

        self.det_norm = nn.LayerNorm(feature_dim)
        self.cap_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        g1_dict: Dict[int, torch.Tensor],
        g2_dict: Dict[int, torch.Tensor],
        layer_keys: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        det_feats, cap_feats = [], []

        for level_idx, key in enumerate(layer_keys):
            g1 = g1_dict[key]
            g2 = g2_dict[key]

            det_g1 = self.det_gates[level_idx][0](g1)
            det_g2 = self.det_gates[level_idx][1](g2)
            cap_g1 = self.cap_gates[level_idx][0](g1)
            cap_g2 = self.cap_gates[level_idx][1](g2)

            det_feats.append(det_g1)
            det_feats.append(det_g2)
            cap_feats.append(cap_g1)
            cap_feats.append(cap_g2)

        O_det = self.det_norm(self.det_fusion(det_feats))
        O_cap = self.cap_norm(self.cap_fusion(cap_feats))

        return O_det, O_cap
