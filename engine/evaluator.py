from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import compute_all_metrics
from utils.logger import get_logger, MetricTracker


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        cfg,
        device: torch.device,
        split: str = "val",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.split = split
        self.logger = get_logger("Evaluator")

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()

        all_predictions: List[str] = []
        all_references: List[List[str]] = []
        all_pred_maps: List[torch.Tensor] = []
        all_gt_masks: List[torch.Tensor] = []

        encoded_prompt = self.tokenizer.encode_inference()
        prompt_ids = encoded_prompt["input_ids"].unsqueeze(0)
        prompt_mask = encoded_prompt["attention_mask"].unsqueeze(0)

        for batch in tqdm(dataloader, desc=f"Evaluating [{self.split}]"):
            img_a = batch["img_a"].to(self.device, non_blocking=True)
            img_b = batch["img_b"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            references = batch["references"]

            B = img_a.size(0)
            input_ids = prompt_ids.expand(B, -1).to(self.device)
            attention_mask = prompt_mask.expand(B, -1).to(self.device)

            with autocast(enabled=self.cfg.train.amp):
                captions, change_prob = self.model.generate(
                    img_a=img_a,
                    img_b=img_b,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tokenizer=self.tokenizer.tokenizer,
                )

            all_predictions.extend(captions)
            all_references.extend(references)
            all_pred_maps.append(change_prob.cpu())
            all_gt_masks.append(mask.cpu())

        pred_maps = torch.cat(all_pred_maps, dim=0)
        gt_masks = torch.cat(all_gt_masks, dim=0)

        metrics = compute_all_metrics(
            predictions=all_predictions,
            references=all_references,
            pred_maps=pred_maps,
            gt_masks=gt_masks,
        )

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Evaluation Results [{self.split.upper()}]")
        self.logger.info(f"{'='*60}")
        for k, v in sorted(metrics.items()):
            self.logger.info(f"  {k:<15}: {v:.2f}")
        self.logger.info(f"{'='*60}\n")

        return metrics

    @torch.no_grad()
    def evaluate_with_examples(
        self,
        dataloader: DataLoader,
        num_examples: int = 5,
    ) -> Tuple[Dict[str, float], List[Dict]]:
        metrics = self.evaluate(dataloader)

        self.model.eval()
        examples = []
        count = 0

        encoded_prompt = self.tokenizer.encode_inference()
        prompt_ids = encoded_prompt["input_ids"].unsqueeze(0)
        prompt_mask = encoded_prompt["attention_mask"].unsqueeze(0)

        for batch in dataloader:
            if count >= num_examples:
                break

            img_a = batch["img_a"].to(self.device)
            img_b = batch["img_b"].to(self.device)
            references = batch["references"]
            stems = batch["stem"]

            B = img_a.size(0)
            input_ids = prompt_ids.expand(B, -1).to(self.device)
            attention_mask = prompt_mask.expand(B, -1).to(self.device)

            with autocast(enabled=self.cfg.train.amp):
                captions, change_prob = self.model.generate(
                    img_a=img_a,
                    img_b=img_b,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tokenizer=self.tokenizer.tokenizer,
                )

            for i in range(min(B, num_examples - count)):
                examples.append({
                    "stem": stems[i],
                    "prediction": captions[i],
                    "references": references[i],
                    "change_prob_mean": change_prob[i].mean().item(),
                })
                count += 1

        return metrics, examples
