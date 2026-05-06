from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer


class CLIPTextEncoder(nn.Module):
    def __init__(self, backbone: str, output_dim: int, frozen: bool = True):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(backbone)
        self.model = CLIPTextModel.from_pretrained(backbone)
        self.output_dim = output_dim

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def encode(self, texts: List[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        outputs = self.model(**tokens, return_dict=True)
        text_embeds = outputs.pooler_output
        text_embeds = F.normalize(text_embeds, dim=-1)
        return text_embeds

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        return self.encode(texts, device)
