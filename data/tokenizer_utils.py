from typing import List, Dict, Optional, Tuple
import torch
from transformers import AutoTokenizer


PROMPT_TEMPLATE = (
    "You are analyzing a pair of bi-temporal UAV remote sensing images captured "
    "over an urban construction site. The visual tokens encode change-aware "
    "spatial features derived from before and after imagery, supplemented by "
    "detection-guided spatial priors indicating changed regions. Describe the "
    "observed scene changes concisely and accurately.\nChanges: "
)

IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = -200


def build_instruction(caption: Optional[str] = None) -> Tuple[str, Optional[str]]:
    system = PROMPT_TEMPLATE
    return system, caption


class CaptionTokenizer:
    def __init__(self, model_name: str, max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def encode_train(self, caption: str) -> Dict[str, torch.Tensor]:
        system, target = build_instruction(caption)

        prompt_ids = self.tokenizer(
            system,
            add_special_tokens=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        target_ids = self.tokenizer(
            target,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        eos = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        input_ids = torch.cat([prompt_ids, target_ids, eos], dim=0)

        labels = torch.full_like(input_ids, fill_value=-100)
        labels[len(prompt_ids):] = torch.cat([target_ids, eos], dim=0)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def encode_inference(self) -> Dict[str, torch.Tensor]:
        system, _ = build_instruction(caption=None)
        encoded = self.tokenizer(
            system,
            add_special_tokens=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded.input_ids.squeeze(0),
            "attention_mask": encoded.attention_mask.squeeze(0),
        }

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def batch_encode_train(self, captions: List[str]) -> Dict[str, torch.Tensor]:
        encoded_list = [self.encode_train(c) for c in captions]

        max_len = max(e["input_ids"].size(0) for e in encoded_list)
        max_len = min(max_len, self.max_length)

        input_ids_padded = []
        attention_mask_padded = []
        labels_padded = []

        for enc in encoded_list:
            pad_len = max_len - enc["input_ids"].size(0)
            pad_ids = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            pad_mask = torch.zeros(pad_len, dtype=torch.long)
            pad_labels = torch.full((pad_len,), -100, dtype=torch.long)

            input_ids_padded.append(torch.cat([enc["input_ids"], pad_ids]))
            attention_mask_padded.append(torch.cat([enc["attention_mask"], pad_mask]))
            labels_padded.append(torch.cat([enc["labels"], pad_labels]))

        return {
            "input_ids": torch.stack(input_ids_padded),
            "attention_mask": torch.stack(attention_mask_padded),
            "labels": torch.stack(labels_padded),
        }

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id
