from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vision_encoder import CLIPViTEncoder, build_vision_encoder
from models.prototype_bank import PrototypeBank
from models.pg_cai import PGCAI
from models.tamg import TAMG
from models.fpn import FPN
from models.mask_encoder import MaskEncoder
from models.caption_decoder import CaptionDecoder
from models.clip_text_encoder import CLIPTextEncoder


class PTNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layer_keys = list(cfg.vision_encoder.extract_layers)
        self.num_levels = len(self.layer_keys)

        self.vision_encoder = build_vision_encoder(cfg)
        N = self.vision_encoder.get_num_patches()
        D = self.vision_encoder.get_hidden_dim()

        self.prototype_bank = PrototypeBank(
            num_clusters=cfg.prototype.num_clusters,
            spatial_tokens=N,
            feature_dim=D,
            temperature=cfg.prototype.temperature,
        )

        self.pg_cai = PGCAI(
            feature_dim=D,
            num_heads=cfg.pg_cai.num_heads,
            ffn_dim=cfg.pg_cai.ffn_dim,
            num_layers=cfg.pg_cai.num_layers,
            num_levels=self.num_levels,
            dropout=cfg.pg_cai.dropout,
        )

        self.tamg = TAMG(
            feature_dim=D,
            num_heads=cfg.tamg.num_heads,
            num_levels=self.num_levels,
            num_temporal=cfg.tamg.num_temporal,
            gate_dropout=cfg.tamg.gate_dropout,
        )

        self.fpn = FPN(
            in_channels=cfg.fpn.in_channels,
            out_channels=cfg.fpn.out_channels,
            num_levels=self.num_levels,
            output_size=cfg.fpn.output_size,
            det_head_channels=cfg.fpn.det_head_channels,
            spatial_tokens=N,
        )

        self.mask_encoder = MaskEncoder(
            input_channels=cfg.fpn.out_channels,
            pool_size=cfg.mask_encoder.pool_size,
            hidden_dim=cfg.mask_encoder.hidden_dim,
            output_dim=cfg.caption_decoder.llm_dim,
            num_mlp_layers=cfg.mask_encoder.num_mlp_layers,
        )

        self.caption_decoder = CaptionDecoder(
            llm_name=cfg.caption_decoder.llm_name,
            visual_dim=D,
            lora_rank=cfg.caption_decoder.lora_rank,
            lora_alpha=cfg.caption_decoder.lora_alpha,
            lora_dropout=cfg.caption_decoder.lora_dropout,
            lora_target_modules=cfg.caption_decoder.lora_target_modules,
            vl_projector_hidden=cfg.caption_decoder.vl_projector_hidden,
            vl_projector_layers=cfg.caption_decoder.vl_projector_layers,
            max_new_tokens=cfg.caption_decoder.max_new_tokens,
            freeze_except_lora=cfg.caption_decoder.freeze_except_lora,
        )

        self.clip_text_encoder = CLIPTextEncoder(
            backbone=cfg.clip_text.backbone,
            output_dim=cfg.clip_text.output_dim,
            frozen=cfg.clip_text.frozen,
        )

        llm_dim = self.caption_decoder.get_llm_dim()
        self.align_proj = nn.Linear(llm_dim, cfg.clip_text.output_dim, bias=False)

    def _extract_features(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
    ) -> Tuple[Dict, Dict]:
        feats_1 = self.vision_encoder(img_a)
        feats_2 = self.vision_encoder(img_b)
        return feats_1, feats_2

    def forward(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        captions: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        feats_1, feats_2 = self._extract_features(img_a, img_b)

        g1_dict, g2_dict = self.pg_cai(
            feats_1, feats_2, self.prototype_bank, self.layer_keys
        )

        O_det, O_cap = self.tamg(g1_dict, g2_dict, self.layer_keys)

        level_feats = [g1_dict[k] for k in self.layer_keys]
        change_map, fpn_finest = self.fpn(O_det, level_feats)

        det_tokens = self.mask_encoder(fpn_finest)

        dec_out = self.caption_decoder(
            O_cap=O_cap,
            det_tokens=det_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        outputs = {
            "change_map": change_map,
            "caption_loss": dec_out.get("caption_loss"),
            "pooled_hidden": dec_out["pooled_hidden"],
        }

        if captions is not None:
            with torch.no_grad():
                text_embeds = self.clip_text_encoder(captions, img_a.device)
            # visual_embeds = F.normalize(self.align_proj(dec_out["pooled_hidden"]), dim=-1)
            visual_embeds = F.normalize(self.align_proj(dec_out["pooled_hidden"].float()), dim=-1)
            outputs["visual_embeds"] = visual_embeds
            outputs["text_embeds"] = text_embeds

        return outputs

    @torch.no_grad()
    def generate(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer,
    ) -> Tuple[List[str], torch.Tensor]:
        feats_1, feats_2 = self._extract_features(img_a, img_b)

        g1_dict, g2_dict = self.pg_cai(
            feats_1, feats_2, self.prototype_bank, self.layer_keys
        )

        O_det, O_cap = self.tamg(g1_dict, g2_dict, self.layer_keys)

        level_feats = [g1_dict[k] for k in self.layer_keys]
        change_map, fpn_finest = self.fpn(O_det, level_feats)

        det_tokens = self.mask_encoder(fpn_finest)

        captions = self.caption_decoder.generate(
            O_cap=O_cap,
            det_tokens=det_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
        )

        change_prob = torch.sigmoid(change_map)

        return captions, change_prob

    def load_prototype_bank(self, path: str):
        self.prototype_bank.load_pretrained(path)

    def get_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_model(cfg) -> PTNet:
    model = PTNet(cfg)
    if cfg.prototype.save_path:
        try:
            model.load_prototype_bank(cfg.prototype.save_path)
            print(f"Loaded prototype bank from {cfg.prototype.save_path}")
        except FileNotFoundError:
            print("Prototype bank not found, using random initialization.")
    return model
