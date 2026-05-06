import os
import json
import argparse
from pathlib import Path
from typing import Dict, List

import torch

from config import get_config
from data import build_dataloader, CaptionTokenizer
from models import build_model
from utils import (
    seed_everything,
    load_checkpoint,
    get_logger,
    compute_all_metrics,
)
from engine import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="PTNet Evaluation")
    parser.add_argument("--dataset", type=str, default="uccd",
                        choices=["uccd", "whu_cdc"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/test_results")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--num_examples", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = get_config(args.dataset)
    cfg.train.val_batch_size = args.batch_size
    cfg.data.num_workers = args.num_workers
    if args.img_size:
        cfg.data.img_size = args.img_size
        cfg.fpn.output_size = args.img_size
        cfg.__post_init__()

    seed_everything(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = get_logger("test", log_file=str(Path(args.output_dir) / "test.log"))

    logger.info(f"Dataset : {args.dataset}")
    logger.info(f"Split   : {args.split}")
    logger.info(f"Ckpt    : {args.checkpoint}")
    logger.info(f"Device  : {device}")

    test_loader = build_dataloader(
        root=cfg.data.root,
        json_path=cfg.data.json_path,
        split=args.split,
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
    load_checkpoint(args.checkpoint, model)
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
        split=args.split,
    )

    metrics, examples = evaluator.evaluate_with_examples(
        test_loader, num_examples=args.num_examples
    )

    results = {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "metrics": metrics,
    }

    results_path = Path(args.output_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    if args.save_predictions:
        examples_path = Path(args.output_dir) / "predictions.json"
        with open(examples_path, "w") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        logger.info(f"Predictions saved to {examples_path}")

    logger.info("\nFinal Results:")
    for k, v in sorted(metrics.items()):
        logger.info(f"  {k:<15}: {v:.2f}")

    logger.info("\nSample Predictions:")
    for i, ex in enumerate(examples[:5]):
        logger.info(f"\n  [{i+1}] {ex['stem']}")
        logger.info(f"  Pred : {ex['prediction']}")
        logger.info(f"  Refs : {ex['references'][0]}")


if __name__ == "__main__":
    main()
