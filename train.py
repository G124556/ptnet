import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from config import get_config
from data import build_dataloader, CaptionTokenizer
from models import build_model
from utils import (
    PTNetLoss,
    WarmupCosineScheduler,
    MetricLogger,
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    get_parameter_groups,
    count_parameters,
    is_main_process,
    setup_distributed,
    get_rank,
    get_logger,
)
from engine import Trainer, Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="PTNet Training")
    parser.add_argument("--dataset", type=str, default="uccd",
                        choices=["uccd", "whu_cdc"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--prototype_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    is_distributed = setup_distributed()

    cfg = get_config(args.dataset)

    if args.output_dir:
        cfg.train.output_dir = args.output_dir
    if args.resume:
        cfg.train.resume = args.resume
    if args.prototype_path:
        cfg.prototype.save_path = args.prototype_path
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.img_size:
        cfg.data.img_size = args.img_size
        cfg.fpn.output_size = args.img_size
        cfg.__post_init__()
    if args.num_workers:
        cfg.data.num_workers = args.num_workers
    if args.no_wandb:
        cfg.log.use_wandb = False

    seed_everything(cfg.train.seed + get_rank())
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
                          if torch.cuda.is_available() else "cpu")

    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)
    logger = get_logger("train", log_file=str(Path(cfg.train.output_dir) / "train.log"))

    metric_logger = None
    if is_main_process():
        metric_logger = MetricLogger(
            output_dir=cfg.train.output_dir,
            use_wandb=cfg.log.use_wandb,
            use_tensorboard=cfg.log.use_tensorboard,
        )
        metric_logger.init_wandb(
            project=cfg.log.wandb_project,
            name=cfg.experiment_name,
            config=cfg,
            entity=cfg.log.wandb_entity,
        )

    train_loader = build_dataloader(
        root=cfg.data.root,
        json_path=cfg.data.json_path,
        split="train",
        img_size=cfg.data.img_size,
        mean=cfg.data.mean,
        std=cfg.data.std,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        distributed=is_distributed,
    )

    val_loader = build_dataloader(
        root=cfg.data.root,
        json_path=cfg.data.json_path,
        split="val",
        img_size=cfg.data.img_size,
        mean=cfg.data.mean,
        std=cfg.data.std,
        batch_size=cfg.train.val_batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        distributed=False,
    )

    tokenizer = CaptionTokenizer(
        model_name=cfg.caption_decoder.llm_name,
        max_length=128,
    )

    model = build_model(cfg).to(device)

    if is_main_process():
        param_info = count_parameters(model)
        logger.info(
            f"Model parameters — "
            f"Total: {param_info['total']/1e6:.2f}M  "
            f"Trainable: {param_info['trainable']/1e6:.2f}M  "
            f"Frozen: {param_info['frozen']/1e6:.2f}M"
        )

    if is_distributed:
        model = DDP(
            model,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            find_unused_parameters=cfg.dist.find_unused_parameters,
        )

    param_groups = get_parameter_groups(
        model.module if is_distributed else model, cfg
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
    )

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=cfg.scheduler.warmup_epochs,
        total_epochs=cfg.train.epochs,
        min_lr=cfg.scheduler.min_lr,
        warmup_lr_init=cfg.scheduler.warmup_lr_init,
    )

    criterion = PTNetLoss(
        lambda_caption=cfg.loss.lambda_caption,
        lambda_detection=cfg.loss.lambda_detection,
        lambda_align=cfg.loss.lambda_align,
        dynamic_weight_temp=cfg.loss.dynamic_weight_temp,
        bce_pos_weight=cfg.loss.bce_pos_weight,
        infonce_temperature=cfg.loss.infonce_temperature,
    ).to(device)

    start_epoch = 0
    best_cider = 0.0

    if cfg.train.resume:
        start_epoch, best_cider = load_checkpoint(
            cfg.train.resume,
            model.module if is_distributed else model,
            optimizer,
            scheduler,
        )
        logger.info(f"Resumed from epoch {start_epoch}, best CIDEr-D: {best_cider:.2f}")

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
        metric_logger=metric_logger,
    )

    evaluator = Evaluator(
        model=model.module if is_distributed else model,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
        split="val",
    )

    logger.info(f"Starting training for {cfg.train.epochs} epochs")

    for epoch in range(start_epoch, cfg.train.epochs):
        if is_distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_metrics = trainer.train_one_epoch(train_loader, epoch)
        scheduler.step()

        if is_main_process() and (epoch + 1) % cfg.train.eval_freq == 0:
            val_metrics = evaluator.evaluate(val_loader)

            if metric_logger is not None:
                metric_logger.log(
                    {f"val/{k}": v for k, v in val_metrics.items()},
                    step=epoch,
                )
                metric_logger.log_metrics_to_file(
                    {"epoch": epoch, "train": train_metrics, "val": val_metrics}
                )

            cider = val_metrics.get("CIDEr-D", 0.0)
            is_best = cider > best_cider
            if is_best:
                best_cider = cider
                logger.info(f"New best CIDEr-D: {best_cider:.2f} at epoch {epoch+1}")

            m = model.module if is_distributed else model
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model": m.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_metric": best_cider,
                    "val_metrics": val_metrics,
                },
                output_dir=cfg.train.output_dir,
                filename=f"checkpoint_epoch_{epoch+1:03d}.pt",
                is_best=is_best,
            )

        if is_main_process() and (epoch + 1) % cfg.train.save_freq == 0:
            m = model.module if is_distributed else model
            save_checkpoint(
                {"epoch": epoch + 1, "model": m.state_dict()},
                output_dir=cfg.train.output_dir,
                filename="checkpoint_latest.pt",
            )

    if metric_logger is not None:
        metric_logger.close()

    logger.info(f"Training complete. Best CIDEr-D: {best_cider:.2f}")


if __name__ == "__main__":
    main()
