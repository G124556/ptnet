import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class LoRALinearLLM(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: int, dropout: float = 0.05):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / rank

        self.lora_a = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_b = nn.Parameter(torch.zeros(linear.out_features, rank))
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return self.linear(x) + self.dropout(x) @ self.lora_a.T @ self.lora_b.T * self.scaling
        return self.linear(x) + self.dropout(x) @ self.lora_a.T.to(x.dtype) @ self.lora_b.T.to(x.dtype) * self.scaling


def inject_lora_llm(
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
                setattr(parent, child_name, LoRALinearLLM(module, rank, alpha, dropout))
    return model


class VLProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.proj = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class CaptionDecoder(nn.Module):
    def __init__(
        self,
        llm_name: str,
        visual_dim: int,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: Tuple[str, ...],
        vl_projector_hidden: int,
        vl_projector_layers: int,
        max_new_tokens: int = 64,
        freeze_except_lora: bool = True,
    ):
        super().__init__()
        self.max_new_tokens = max_new_tokens

        config = AutoConfig.from_pretrained(llm_name, trust_remote_code=True)
        self.llm_dim = config.hidden_size

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        if freeze_except_lora:
            for param in self.llm.parameters():
                param.requires_grad = False

        inject_lora_llm(self.llm, lora_target_modules, lora_rank, lora_alpha, lora_dropout)

        self.vl_projector = VLProjector(
            input_dim=visual_dim,
            hidden_dim=vl_projector_hidden,
            output_dim=self.llm_dim,
            num_layers=vl_projector_layers,
        )

    def _build_visual_prefix(
        self,
        O_cap: torch.Tensor,
        det_tokens: torch.Tensor,
    ) -> torch.Tensor:
        cap_projected = self.vl_projector(O_cap)
        visual_tokens = torch.cat([det_tokens, cap_projected], dim=1)
        return visual_tokens

    def forward(
        self,
        O_cap: torch.Tensor,
        det_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        visual_prefix = self._build_visual_prefix(O_cap, det_tokens)
        visual_prefix = visual_prefix.to(self.llm.dtype)

        text_embeds = self.llm.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([visual_prefix, text_embeds], dim=1)

        vis_len = visual_prefix.size(1)
        vis_mask = torch.ones(
            attention_mask.size(0), vis_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

        if labels is not None:
            vis_labels = torch.full(
                (labels.size(0), vis_len), -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            full_labels = torch.cat([vis_labels, labels], dim=1)
        else:
            full_labels = None

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = outputs.hidden_states[-1]
        seq_len = last_hidden.size(1)
        pooled = last_hidden.mean(dim=1)

        result = {
            "logits": outputs.logits,
            "pooled_hidden": pooled,
        }

        if outputs.loss is not None:
            result["caption_loss"] = outputs.loss

        return result

    @torch.no_grad()
    def generate(
        self,
        O_cap: torch.Tensor,
        det_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer,
    ) -> List[str]:
        visual_prefix = self._build_visual_prefix(O_cap, det_tokens)
        visual_prefix = visual_prefix.to(self.llm.dtype)

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_prefix, text_embeds], dim=1)

        vis_len = visual_prefix.size(1)
        vis_mask = torch.ones(
            attention_mask.size(0), vis_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_mask = torch.cat([vis_mask, attention_mask], dim=1)

        generated = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        return [tokenizer.decode(g, skip_special_tokens=True) for g in generated]

    def get_llm_dim(self) -> int:
        return self.llm_dim
