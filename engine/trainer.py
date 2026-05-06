from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from utils.losses import PTNetLoss
from utils.logger import MetricTracker, get_logger
from utils.misc import clip_gradients, is_main_process


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: PTNetLoss,
        optimizer: torch.optim.Optimizer,
        scheduler,
        tokenizer,
        cfg,
        device: torch.device,
        metric_logger=None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.metric_logger = metric_logger
        self.logger = get_logger("Trainer")

        self.scaler = GradScaler(enabled=cfg.train.amp)
        self.accum_steps = cfg.optimizer.gradient_accumulation_steps
        self.global_step = 0

    def train_one_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        tracker = MetricTracker()
        self.optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            img_a = batch["img_a"].to(self.device, non_blocking=True)
            img_b = batch["img_b"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            captions = batch["caption"]

            encoded = self.tokenizer.batch_encode_train(captions)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            labels = encoded["labels"].to(self.device)

            with autocast(enabled=self.cfg.train.amp):
                outputs = self.model(
                    img_a=img_a,
                    img_b=img_b,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    captions=captions,
                )

                loss_dict = self.criterion(outputs, mask)
                loss = loss_dict["total_loss"] / self.accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = clip_gradients(self.model, self.cfg.optimizer.gradient_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

                log_metrics = {
                    "train/total_loss": loss_dict["total_loss"].item(),
                    "train/caption_loss": loss_dict["caption_loss"].item(),
                    "train/detection_loss": loss_dict["detection_loss"].item(),
                    "train/alignment_loss": loss_dict["alignment_loss"].item(),
                    "train/lambda_c": loss_dict["lambda_c"],
                    "train/lambda_d": loss_dict["lambda_d"],
                    "train/grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                    "train/lr": self.optimizer.param_groups[-1]["lr"],
                }

                tracker.update({k.split("/")[-1]: v for k, v in log_metrics.items()})

                if is_main_process() and self.global_step % self.cfg.train.log_freq == 0:
                    if self.metric_logger is not None:
                        self.metric_logger.log(log_metrics, step=self.global_step)

                    avgs = tracker.averages()
                    self.logger.info(
                        f"Epoch [{epoch}] Step [{self.global_step}] "
                        f"Loss: {avgs.get('total_loss', 0):.4f} "
                        f"Lc: {avgs.get('caption_loss', 0):.4f} "
                        f"Ld: {avgs.get('detection_loss', 0):.4f} "
                        f"La: {avgs.get('alignment_loss', 0):.4f} "
                        f"LR: {self.optimizer.param_groups[-1]['lr']:.2e}"
                    )

        return tracker.averages()
