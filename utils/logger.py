import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_file is not None:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


class MetricLogger:
    def __init__(self, output_dir: str, use_wandb: bool = False, use_tensorboard: bool = True):
        self.output_dir = Path(output_dir)
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self._tb_writer = None
        self._step = 0

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.output_dir / "tensorboard"
                tb_dir.mkdir(parents=True, exist_ok=True)
                self._tb_writer = SummaryWriter(str(tb_dir))
            except ImportError:
                pass

    def init_wandb(self, project: str, name: str, config: Any, entity: Optional[str] = None):
        if not self.use_wandb:
            return
        try:
            import wandb
            wandb.init(project=project, name=name, config=config, entity=entity)
        except ImportError:
            self.use_wandb = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, prefix: str = ""):
        s = step if step is not None else self._step
        self._step = s + 1

        tagged = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}

        if self.use_wandb:
            try:
                import wandb
                wandb.log(tagged, step=s)
            except Exception:
                pass

        if self._tb_writer is not None:
            for k, v in tagged.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, s)

    def log_metrics_to_file(self, metrics: Dict[str, Any], filename: str = "metrics.json"):
        path = self.output_dir / filename
        existing = []
        if path.exists():
            with open(path) as f:
                existing = json.load(f)
        existing.append(metrics)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)

    def close(self):
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.avg:.4f}"


class MetricTracker:
    def __init__(self):
        self._meters: Dict[str, AverageMeter] = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            if k not in self._meters:
                self._meters[k] = AverageMeter()
            self._meters[k].update(float(v), n)

    def averages(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self._meters.items()}

    def reset(self):
        for m in self._meters.values():
            m.reset()
