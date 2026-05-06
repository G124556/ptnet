import os
import sys
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from data.dataset import PrototypeBuildDataset
from models.vision_encoder import build_vision_encoder


def compute_rbf_expansion(
    Z: torch.Tensor,
    changed_indices: torch.Tensor,
    sigma: float,
    grid_size: int,
) -> torch.Tensor:
    N, D = Z.shape
    device = Z.device

    row = torch.arange(grid_size, device=device).float()
    col = torch.arange(grid_size, device=device).float()
    grid_y, grid_x = torch.meshgrid(row, col, indexing="ij")
    coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)

    if changed_indices.numel() == 0:
        return Z.unsqueeze(0).expand(N, -1, -1).mean(dim=1)

    src_coords = coords[changed_indices]
    src_Z = Z[changed_indices]

    dists = torch.cdist(coords, src_coords)
    weights = torch.exp(-dists ** 2 / (2 * sigma ** 2))
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

    Z_expanded = weights @ src_Z
    return Z_expanded


@torch.no_grad()
def extract_difference_features(
    vision_encoder: torch.nn.Module,
    dataloader: DataLoader,
    extract_layer: int,
    device: torch.device,
) -> tuple:
    all_z = []
    all_masks = []

    vision_encoder.eval()

    for batch in tqdm(dataloader, desc="Extracting features"):
        img_a = batch["img_a"].to(device)
        img_b = batch["img_b"].to(device)
        mask = batch["mask"].to(device)

        feats_1 = vision_encoder(img_a)
        feats_2 = vision_encoder(img_b)

        F1 = feats_1[extract_layer]
        F2 = feats_2[extract_layer]

        Z = torch.abs(F1 - F2)

        B, N, D = Z.shape
        grid_size = int(math.isqrt(N))

        mask_resized = F.interpolate(
            mask.float().unsqueeze(1),
            size=(grid_size, grid_size),
            mode="nearest",
        ).squeeze(1).reshape(B, N)

        for b in range(B):
            z_b = Z[b]
            m_b = mask_resized[b]
            changed_idx = (m_b > 0).nonzero(as_tuple=True)[0]

            if changed_idx.numel() > 0:
                z_pooled = z_b[changed_idx].mean(dim=0)
            else:
                z_pooled = z_b.mean(dim=0)

            all_z.append(z_pooled.cpu())
            all_masks.append(m_b.cpu())

    return torch.stack(all_z, dim=0), torch.stack(all_masks, dim=0)


@torch.no_grad()
def build_prototype_bank(
    vision_encoder: torch.nn.Module,
    dataloader: DataLoader,
    extract_layer: int,
    num_clusters: int,
    sigma: float,
    temperature: float,
    device: torch.device,
    kmeans_seed: int = 42,
    kmeans_iters: int = 300,
) -> torch.Tensor:
    all_z, all_masks = extract_difference_features(
        vision_encoder, dataloader, extract_layer, device
    )

    z_np = all_z.numpy()
    kmeans = KMeans(
        n_clusters=num_clusters,
        max_iter=kmeans_iters,
        random_state=kmeans_seed,
        n_init=10,
    )
    kmeans.fit(z_np)
    cluster_centers = torch.from_numpy(kmeans.cluster_centers_).float()
    labels = torch.from_numpy(kmeans.labels_).long()

    print(f"\nK-means cluster distribution:")
    for k in range(num_clusters):
        count = (labels == k).sum().item()
        print(f"  Cluster {k}: {count} samples ({100*count/len(labels):.1f}%)")

    N = all_masks.shape[1]
    D = all_z.shape[1]
    grid_size = int(math.isqrt(N))

    print(f"\nBuilding prototype bank: K={num_clusters}, N={N}, D={D}")

    prototypes = torch.zeros(num_clusters, N, D)
    proto_weights = torch.zeros(num_clusters)

    vision_encoder.eval()
    sample_idx = 0
    dataset = dataloader.dataset

    for batch in tqdm(dataloader, desc="Computing prototypes"):
        img_a = batch["img_a"].to(device)
        img_b = batch["img_b"].to(device)
        mask = batch["mask"].to(device)

        feats_1 = vision_encoder(img_a)
        feats_2 = vision_encoder(img_b)

        F1 = feats_1[extract_layer]
        F2 = feats_2[extract_layer]
        Z_batch = torch.abs(F1 - F2)

        B = img_a.size(0)
        mask_resized = F.interpolate(
            mask.float().unsqueeze(1),
            size=(grid_size, grid_size),
            mode="nearest",
        ).squeeze(1).reshape(B, N)

        for b in range(B):
            z_b = Z_batch[b].cpu()
            m_b = mask_resized[b].cpu()
            z_scalar = all_z[sample_idx]

            changed_idx = (m_b > 0).nonzero(as_tuple=True)[0]
            Z_expanded = compute_rbf_expansion(z_b, changed_idx, sigma, grid_size)

            dists = torch.norm(
                z_scalar.unsqueeze(0) - cluster_centers, dim=-1
            )
            soft_assign = F.softmax(-dists / temperature, dim=0)

            for k in range(num_clusters):
                prototypes[k] += soft_assign[k].item() * Z_expanded
                proto_weights[k] += soft_assign[k].item()

            sample_idx += 1

    for k in range(num_clusters):
        if proto_weights[k] > 0:
            prototypes[k] /= proto_weights[k]

    return prototypes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="uccd")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    cfg = get_config(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Building prototype bank for {args.dataset}")
    print(f"  K = {cfg.prototype.num_clusters}")
    print(f"  img_size = {cfg.data.img_size}")
    print(f"  extract_layer = {cfg.vision_encoder.extract_layers[1]}")

    dataset = PrototypeBuildDataset(
        root=cfg.data.root,
        json_path=cfg.data.json_path,
        img_size=cfg.data.img_size,
        mean=cfg.data.mean,
        std=cfg.data.std,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Training set size: {len(dataset)} unique images")

    vision_encoder = build_vision_encoder(cfg).to(device)
    vision_encoder.eval()

    extract_layer = cfg.vision_encoder.extract_layers[1]

    prototypes = build_prototype_bank(
        vision_encoder=vision_encoder,
        dataloader=dataloader,
        extract_layer=extract_layer,
        num_clusters=cfg.prototype.num_clusters,
        sigma=cfg.prototype.rbf_sigma,
        temperature=cfg.prototype.assignment_temperature,
        device=device,
        kmeans_seed=cfg.prototype.kmeans_seed,
        kmeans_iters=cfg.prototype.kmeans_iters,
    )

    save_path = Path(cfg.prototype.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "prototypes": prototypes,
            "num_clusters": cfg.prototype.num_clusters,
            "spatial_tokens": cfg.prototype.spatial_tokens,
            "feature_dim": cfg.vision_encoder.hidden_dim,
            "dataset": args.dataset,
        },
        save_path,
    )

    print(f"\nPrototype bank saved to {save_path}")
    print(f"Shape: {prototypes.shape}")
    print(f"Memory: {prototypes.numel() * 4 / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
