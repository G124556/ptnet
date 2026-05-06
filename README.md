# PTNet: Prototype-Guided Task-Adaptive Network for Remote Sensing Change Captioning

A unified framework for joint change detection and captioning on UAV-based urban construction imagery.


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
python scripts/build_prototypes.py --dataset uccd
```

This runs K-means clustering on training-set difference features from CLIP layer 12, applies RBF spatial interpolation, and saves the prototype bank to `./cache/prototypes_uccd.pt`.

### Step 2 — Train

Single GPU:
```bash
python train.py --dataset uccd
```


Key arguments:
```
--dataset        uccd | whu_cdc
--output_dir     path to save checkpoints and logs
--resume         path to checkpoint to resume from
--batch_size     override batch size
--img_size       override input resolution 
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


