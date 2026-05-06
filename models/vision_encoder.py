import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPVisionConfig


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: int, dropout: float = 0.05):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scaling = alpha / rank

        self.lora_a = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_b = nn.Parameter(torch.zeros(linear.out_features, rank))
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = self.dropout(x) @ self.lora_a.T @ self.lora_b.T
        return base + lora * self.scaling


def inject_lora(
    model: nn.Module,
    target_modules: Tuple[str, ...],
    rank: int,
    alpha: int,
    dropout: float,
) -> nn.Module:
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent_name, child_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout))
    return model


def interpolate_pos_embedding(
    pos_embed: torch.Tensor,
    src_grid: int,
    tgt_grid: int,
) -> torch.Tensor:
    cls_token = pos_embed[:, :1, :]
    patch_embed = pos_embed[:, 1:, :]

    patch_embed = patch_embed.reshape(1, src_grid, src_grid, -1).permute(0, 3, 1, 2)
    patch_embed = F.interpolate(
        patch_embed,
        size=(tgt_grid, tgt_grid),
        mode="bicubic",
        align_corners=False,
    )
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, tgt_grid * tgt_grid, -1)
    return torch.cat([cls_token, patch_embed], dim=1)


class CLIPViTEncoder(nn.Module):
    def __init__(
        self,
        backbone: str,
        img_size: int,
        patch_size: int,
        extract_layers: Tuple[int, ...],
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: Tuple[str, ...],
        freeze_except_lora: bool = True,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.extract_layers = sorted(extract_layers)
        self.tgt_grid = img_size // patch_size
        self.num_patches = self.tgt_grid ** 2

        config = CLIPVisionConfig.from_pretrained(backbone)
        self.src_grid = config.image_size // config.patch_size
        self.hidden_dim = config.hidden_size

        self.model = CLIPVisionModel.from_pretrained(backbone)

        self._interpolate_position_embeddings()

        if freeze_except_lora:
            for param in self.model.parameters():
                param.requires_grad = False

        inject_lora(self.model, lora_target_modules, lora_rank, lora_alpha, lora_dropout)

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    @torch.no_grad()
    def _interpolate_position_embeddings(self):
        if self.src_grid == self.tgt_grid:
            return

        pos_embed = self.model.vision_model.embeddings.position_embedding.weight
        pos_embed = pos_embed.unsqueeze(0)
        new_pos_embed = interpolate_pos_embedding(pos_embed, self.src_grid, self.tgt_grid)
        new_pos_embed = new_pos_embed.squeeze(0)

        new_embedding = nn.Embedding(new_pos_embed.size(0), new_pos_embed.size(1))
        new_embedding.weight = nn.Parameter(new_pos_embed)
        self.model.vision_model.embeddings.position_embedding = new_embedding
        self.model.vision_model.embeddings.position_ids = torch.arange(
            new_pos_embed.size(0)
        ).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        outputs = self.model(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states

        features = {}
        for layer_idx in self.extract_layers:
            feat = hidden_states[layer_idx][:, 1:, :]
            features[layer_idx] = feat

        return features

    def get_num_patches(self) -> int:
        return self.num_patches

    def get_hidden_dim(self) -> int:
        return self.hidden_dim


def build_vision_encoder(cfg) -> CLIPViTEncoder:
    return CLIPViTEncoder(
        backbone=cfg.vision_encoder.backbone,
        img_size=cfg.data.img_size,
        patch_size=cfg.vision_encoder.patch_size,
        extract_layers=cfg.vision_encoder.extract_layers,
        lora_rank=cfg.vision_encoder.lora_rank,
        lora_alpha=cfg.vision_encoder.lora_alpha,
        lora_dropout=cfg.vision_encoder.lora_dropout,
        lora_target_modules=cfg.vision_encoder.lora_target_modules,
        freeze_except_lora=cfg.vision_encoder.freeze_except_lora,
        gradient_checkpointing=cfg.vision_encoder.gradient_checkpointing,
    )
