# PTNet: Prototype-Guided Task-Adaptive Network for Remote Sensing Change Captioning

A unified framework for joint change detection and captioning on UAV-based urban construction imagery.

---

## Project Structure

```
PTNet/
├── config.py                     # All hyperparameters (dataclass-based)
├── train.py                      # Training entry point
├── test.py                       # Evaluation entry point
├── requirements.txt
│
├── data/
│   ├── dataset.py                # UCCDDataset, PrototypeBuildDataset, dataloaders
│   ├── transforms.py             # Synchronized augmentation pipeline
│   ├── tokenizer_utils.py        # Instruction formatting for Qwen2
│   └── __init__.py
│
├── models/
│   ├── ptnet.py                  # Full model assembly
│   ├── vision_encoder.py         # CLIP ViT-L/14 + positional encoding interpolation + LoRA
│   ├── prototype_bank.py         # Learnable prototype bank (K × N × D)
│   ├── pg_cai.py                 # Prototype-Guided Cross-Aware Interaction
│   ├── tamg.py                   # Task-Adaptive Multi-head Gating
│   ├── fpn.py                    # FPN-based change detection decoder
│   ├── mask_encoder.py           # FPN feature → detection tokens
│   ├── caption_decoder.py        # Qwen2-1.5B-Instruct + LoRA + VL Projector
│   ├── clip_text_encoder.py      # Frozen CLIP text encoder for alignment loss
│   └── __init__.py
│
├── utils/
│   ├── losses.py                 # DetectionLoss, AlignmentLoss, DynamicWeightBalancer
│   ├── metrics.py                # BLEU-1~4, METEOR, ROUGE-L, CIDEr-D, F1, IoU
│   ├── lr_scheduler.py           # Warmup + cosine decay
│   ├── logger.py                 # WandB + TensorBoard + AverageMeter
│   ├── misc.py                   # Distributed utils, checkpoint, param groups
│   └── __init__.py
│
├── engine/
│   ├── trainer.py                # Training loop with AMP + gradient accumulation
│   ├── evaluator.py              # Evaluation loop with beam search generation
│   └── __init__.py
│
└── scripts/
    ├── build_prototypes.py       # Offline K-means + RBF prototype initialization
    └── __init__.py
```

---

## Data Format

```
split_3_images/
├── train/
│   ├── A/          # Pre-change images
│   ├── B/          # Post-change images
│   └── Label/      # Binary change masks
├── val/
└── test/

wanzhengbanbe.json  # 5 captions per image pair from 5 different VLMs
```

Each image pair expands to **5 independent training samples** (one per caption). At evaluation, a single prediction is scored against all 5 references.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Build Prototype Bank (run once before training)

```bash
python scripts/build_prototypes.py --dataset uccd --batch_size 16
```

This runs K-means clustering on training-set difference features from CLIP layer 12, applies RBF spatial interpolation, and saves the prototype bank to `./cache/prototypes_uccd.pt`.

### Step 2 — Train

Single GPU:
```bash
python train.py --dataset uccd
```

Multi-GPU (2×A100):
```bash
torchrun --nproc_per_node=2 train.py --dataset uccd
```

Key arguments:
```
--dataset        uccd | whu_cdc
--output_dir     path to save checkpoints and logs
--resume         path to checkpoint to resume from
--batch_size     override batch size
--img_size       override input resolution (default: 512)
--no_wandb       disable wandb logging
```

### Step 3 — Test

```bash
python test.py \
    --dataset uccd \
    --checkpoint outputs/ptnet_uccd/best_model.pt \
    --split test \
    --save_predictions
```

---

## Key Design Choices

**Prototype Bank** — Initialized offline via K-means (K=8 for UCCD, K=5 for WHU-CDC) on masked difference features from the CLIP 12th layer. Treated as a learnable `nn.Parameter` and jointly optimized during training.

**Position Encoding Interpolation** — CLIP ViT-L/14 default resolution is 224. For 512×512 input, patch embeddings are bicubically interpolated from a 16×16 grid to a 36×36 grid once at model initialization and cached as fixed weights.

**Training Samples** — Each image pair has 5 captions from different VLMs. Training flat-expands to 5 samples per pair; evaluation uses all 5 as references for a single prediction.

**Dynamic Loss Weighting** — λ_c and λ_d are adapted each epoch based on each task's loss improvement ratio, assigning higher weight to the slower-improving task.

---

## Hyperparameters

| Parameter | UCCD | WHU-CDC |
|-----------|------|---------|
| img_size | 512 | 512 |
| num_prototypes K | 8 | 5 |
| ViT LoRA rank | 16 | 16 |
| LLM LoRA rank | 64 | 64 |
| batch_size | 8 | 8 |
| epochs | 200 | 200 |
| optimizer | AdamW | AdamW |
| base lr | 1e-4 | 1e-4 |
| ViT / LLM lr | 1e-5 | 1e-5 |
| weight decay | 5e-4 | 5e-4 |
| warmup epochs | 5 | 5 |
# ptnet
