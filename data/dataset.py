import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from data.transforms import build_transform, ChangeDetectionTransform


class UCCDDataset(Dataset):
    def __init__(
        self,
        root: str,
        json_path: str,
        split: str,
        transform: ChangeDetectionTransform,
        training: bool = True,
    ):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.root = Path(root)
        self.split = split
        self.training = training
        self.transform = transform

        self.img_a_dir = self.root / split / "A"
        self.img_b_dir = self.root / split / "B"
        self.mask_dir = self.root / split / "Label"

        with open(json_path, "r") as f:
            raw = json.load(f)

        all_items = [item for item in raw["images"] if item["split"] == split]

        if training:
            self.samples = self._build_train_samples(all_items)
        else:
            self.samples = self._build_eval_samples(all_items)

    def _build_train_samples(self, items: List[Dict]) -> List[Dict]:
        samples = []
        for item in items:
            stem = Path(item["filename"]).stem
            for sent in item["sentences"]:
                samples.append({
                    "stem": stem,
                    "imgid": item["imgid"],
                    "caption": sent["raw"],
                    "model_name": sent["model_name"],
                })
        return samples

    def _build_eval_samples(self, items: List[Dict]) -> List[Dict]:
        samples = []
        for item in items:
            stem = Path(item["filename"]).stem
            samples.append({
                "stem": stem,
                "imgid": item["imgid"],
                "references": [sent["raw"] for sent in item["sentences"]],
            })
        return samples

    def _load_image(self, directory: Path, stem: str) -> Image.Image:
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            path = directory / (stem + ext)
            if path.exists():
                return Image.open(path).convert("RGB")
        raise FileNotFoundError(f"No image found for stem '{stem}' in {directory}")

    def _load_mask(self, stem: str) -> Image.Image:
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            path = self.mask_dir / (stem + ext)
            if path.exists():
                return Image.open(path).convert("L")
        raise FileNotFoundError(f"No mask found for stem '{stem}' in {self.mask_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        stem = sample["stem"]

        img_a = self._load_image(self.img_a_dir, stem)
        img_b = self._load_image(self.img_b_dir, stem)
        mask = self._load_mask(stem)

        img_a, img_b, mask = self.transform(img_a, img_b, mask)

        if self.training:
            return {
                "img_a": img_a,
                "img_b": img_b,
                "mask": mask,
                "caption": sample["caption"],
                "imgid": sample["imgid"],
                "model_name": sample["model_name"],
                "stem": stem,
            }
        else:
            return {
                "img_a": img_a,
                "img_b": img_b,
                "mask": mask,
                "references": sample["references"],
                "imgid": sample["imgid"],
                "stem": stem,
            }


class PrototypeBuildDataset(Dataset):
    def __init__(self, root: str, json_path: str, img_size: int, mean: Tuple, std: Tuple):
        self.root = Path(root)
        self.transform = build_transform(img_size, mean, std, training=False)

        with open(json_path, "r") as f:
            raw = json.load(f)

        seen = set()
        self.samples = []
        for item in raw["images"]:
            if item["split"] == "train" and item["imgid"] not in seen:
                seen.add(item["imgid"])
                self.samples.append({
                    "stem": Path(item["filename"]).stem,
                    "imgid": item["imgid"],
                })

        self.img_a_dir = self.root / "train" / "A"
        self.img_b_dir = self.root / "train" / "B"
        self.mask_dir = self.root / "train" / "Label"

    def _load_image(self, directory: Path, stem: str) -> Image.Image:
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            path = directory / (stem + ext)
            if path.exists():
                return Image.open(path).convert("RGB")
        raise FileNotFoundError(f"No image found for stem '{stem}' in {directory}")

    def _load_mask(self, stem: str) -> Image.Image:
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            path = self.mask_dir / (stem + ext)
            if path.exists():
                return Image.open(path).convert("L")
        raise FileNotFoundError(f"No mask found for stem '{stem}' in {self.mask_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        stem = sample["stem"]

        img_a = self._load_image(self.img_a_dir, stem)
        img_b = self._load_image(self.img_b_dir, stem)
        mask = self._load_mask(stem)

        img_a, img_b, mask = self.transform(img_a, img_b, mask)

        return {
            "img_a": img_a,
            "img_b": img_b,
            "mask": mask,
            "imgid": sample["imgid"],
            "stem": stem,
        }


def collate_train(batch: List[Dict]) -> Dict[str, Any]:
    return {
        "img_a": torch.stack([b["img_a"] for b in batch]),
        "img_b": torch.stack([b["img_b"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "caption": [b["caption"] for b in batch],
        "imgid": [b["imgid"] for b in batch],
        "model_name": [b["model_name"] for b in batch],
        "stem": [b["stem"] for b in batch],
    }


def collate_eval(batch: List[Dict]) -> Dict[str, Any]:
    return {
        "img_a": torch.stack([b["img_a"] for b in batch]),
        "img_b": torch.stack([b["img_b"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "references": [b["references"] for b in batch],
        "imgid": [b["imgid"] for b in batch],
        "stem": [b["stem"] for b in batch],
    }


def build_dataloader(
    root: str,
    json_path: str,
    split: str,
    img_size: int,
    mean: Tuple,
    std: Tuple,
    batch_size: int,
    num_workers: int = 8,
    pin_memory: bool = True,
    distributed: bool = False,
) -> DataLoader:
    training = split == "train"
    transform = build_transform(img_size, mean, std, training=training)
    dataset = UCCDDataset(root, json_path, split, transform, training=training)

    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=training)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(training and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=training,
        collate_fn=collate_train if training else collate_eval,
        persistent_workers=(num_workers > 0),
    )
