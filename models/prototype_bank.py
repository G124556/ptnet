from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    def __init__(
        self,
        num_clusters: int,
        spatial_tokens: int,
        feature_dim: int,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.K = num_clusters
        self.N = spatial_tokens
        self.D = feature_dim
        self.temperature = temperature

        self.prototypes = nn.Parameter(
            torch.randn(num_clusters, spatial_tokens, feature_dim)
        )
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def load_pretrained(self, path: str):
        state = torch.load(path, map_location="cpu")
        proto = state["prototypes"]
        if proto.shape != self.prototypes.shape:
            raise ValueError(
                f"Prototype shape mismatch: expected {self.prototypes.shape}, got {proto.shape}"
            )
        self.prototypes.data.copy_(proto)

    def get_soft_assignments(self, z: torch.Tensor) -> torch.Tensor:
        centers = self.prototypes.mean(dim=1)
        dists = torch.cdist(z.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)
        return F.softmax(-dists / self.temperature, dim=-1)

    def forward(self) -> torch.Tensor:
        return self.prototypes
