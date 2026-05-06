import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_dist_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_available():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist_available():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(backend: str = "nccl", dist_url: str = "env://"):
    if "RANK" not in os.environ:
        return False
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method=dist_url,
                            world_size=world_size, rank=rank)
    dist.barrier()
    return True


def save_checkpoint(
    state: Dict,
    output_dir: str,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
):
    path = Path(output_dir) / filename
    torch.save(state, path)
    if is_best:
        best_path = Path(output_dir) / "best_model.pt"
        torch.save(state, best_path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)

    start_epoch = state.get("epoch", 0)
    best_metric = state.get("best_metric", 0.0)

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])

    return start_epoch, best_metric


def get_parameter_groups(model: nn.Module, cfg) -> list:
    vit_params, llm_params, other_params = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "vision_encoder" in name:
            vit_params.append(param)
        elif "caption_decoder" in name:
            llm_params.append(param)
        else:
            other_params.append(param)

    return [
        {"params": vit_params, "lr": cfg.optimizer.vision_encoder_lr},
        {"params": llm_params, "lr": cfg.optimizer.llm_lr},
        {"params": other_params, "lr": cfg.optimizer.lr},
    ]


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def clip_gradients(model: nn.Module, max_norm: float):
    return nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_norm,
    )
