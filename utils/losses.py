from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    def __init__(self, pos_weight: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pw = torch.tensor([self.pos_weight], device=pred.device, dtype=pred.dtype)
        return F.binary_cross_entropy_with_logits(
            pred, target.float(), pos_weight=pw
        )


class AlignmentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, visual_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
        visual_embeds = F.normalize(visual_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        sim = visual_embeds @ text_embeds.T / self.temperature

        B = sim.size(0)
        labels = torch.arange(B, device=sim.device)

        loss_v2t = F.cross_entropy(sim, labels)
        loss_t2v = F.cross_entropy(sim.T, labels)

        return (loss_v2t + loss_t2v) / 2.0


class DynamicWeightBalancer:
    def __init__(self, num_tasks: int = 2, temperature: float = 2.0):
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.prev_losses = [None] * num_tasks
        self.prev_prev_losses = [None] * num_tasks

    def get_weights(self, current_losses: Tuple[float, ...]) -> Tuple[float, ...]:
        if any(l is None for l in self.prev_losses):
            self.prev_prev_losses = list(current_losses)
            self.prev_losses = list(current_losses)
            return tuple(1.0 for _ in range(self.num_tasks))

        ws = []
        for k in range(self.num_tasks):
            if self.prev_prev_losses[k] > 1e-8:
                w = self.prev_losses[k] / (self.prev_prev_losses[k] + 1e-8)
            else:
                w = 1.0
            ws.append(w)

        exp_ws = [torch.tensor(w).exp() for w in ws]
        total = sum(exp_ws)
        weights = [self.num_tasks * e / total for e in exp_ws]

        self.prev_prev_losses = list(self.prev_losses)
        self.prev_losses = list(current_losses)

        return tuple(w.item() if hasattr(w, "item") else w for w in weights)

    def reset(self):
        self.prev_losses = [None] * self.num_tasks
        self.prev_prev_losses = [None] * self.num_tasks


class PTNetLoss(nn.Module):
    def __init__(
        self,
        lambda_caption: float = 1.0,
        lambda_detection: float = 1.0,
        lambda_align: float = 0.3,
        dynamic_weight_temp: float = 2.0,
        bce_pos_weight: float = 2.0,
        infonce_temperature: float = 0.07,
    ):
        super().__init__()
        self.lambda_align = lambda_align

        self.det_loss_fn = DetectionLoss(pos_weight=bce_pos_weight)
        self.align_loss_fn = AlignmentLoss(temperature=infonce_temperature)
        self.balancer = DynamicWeightBalancer(num_tasks=2, temperature=dynamic_weight_temp)

        self._caption_loss_hist = 1.0
        self._det_loss_hist = 1.0

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        mask_gt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        caption_loss = model_outputs["caption_loss"]
        change_map = model_outputs["change_map"]

        det_loss = self.det_loss_fn(change_map, mask_gt)

        lc_val = caption_loss.item()
        ld_val = det_loss.item()

        lambda_c, lambda_d = self.balancer.get_weights((lc_val, ld_val))

        total = lambda_c * caption_loss + lambda_d * det_loss

        if "visual_embeds" in model_outputs and "text_embeds" in model_outputs:
            align_loss = self.align_loss_fn(
                model_outputs["visual_embeds"],
                model_outputs["text_embeds"],
            )
            total = total + self.lambda_align * align_loss
        else:
            align_loss = torch.tensor(0.0, device=change_map.device)

        return {
            "total_loss": total,
            "caption_loss": caption_loss,
            "detection_loss": det_loss,
            "alignment_loss": align_loss,
            "lambda_c": lambda_c,
            "lambda_d": lambda_d,
        }
