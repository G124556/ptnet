from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import os


@dataclass
class DataConfig:
    root: str = "/data2/gaoyupeng/LESPS-master/ALL_data/self/split_3_images"
    json_path: str = "/data2/gaoyupeng/LESPS-master/ALL_data/self/split_3_images/final0429.json"
    img_size: int = 224
    mask_size: int = 224
    mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    num_workers: int = 8
    pin_memory: bool = True
    train_aug: bool = True
    val_aug: bool = False


@dataclass
class VisionEncoderConfig:
    backbone: str = "openai/clip-vit-large-patch14"
    patch_size: int = 14
    hidden_dim: int = 1024
    num_layers: int = 24
    extract_layers: Tuple[int, ...] = (6, 12, 18, 24)
    pretrained_img_size: int = 224
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")
    freeze_except_lora: bool = True
    gradient_checkpointing: bool = True


@dataclass
class PrototypeConfig:
    num_clusters: int = 8
    feature_dim: int = 1024
    temperature: float = 0.07
    assignment_temperature: float = 0.5
    rbf_sigma: float = 1.0
    kmeans_iters: int = 300
    kmeans_seed: int = 42
    save_path: str = "./cache/prototypes_uccd_336.pt"
    spatial_tokens: int = 1296


@dataclass
class PGCAIConfig:
    num_layers: int = 2
    num_heads: int = 16
    dropout: float = 0.1
    feature_dim: int = 1024
    ffn_dim: int = 256


@dataclass
class TAMGConfig:
    num_heads: int = 16
    feature_dim: int = 1024
    num_levels: int = 4
    num_temporal: int = 2
    gate_dropout: float = 0.1


@dataclass
class FPNConfig:
    in_channels: Tuple[int, ...] = (1024, 1024, 1024, 1024)
    out_channels: int = 256
    num_levels: int = 4
    output_size: int = 224
    det_head_channels: int = 128


@dataclass
class MaskEncoderConfig:
    pool_size: Tuple[int, int] = (16, 16)
    input_channels: int = 256
    hidden_dim: int = 512
    # output_dim: int = 1536
    llm_dim: int = 896      # 从1536改成896，0.5B的hidden_dim是896
    num_mlp_layers: int = 2


@dataclass
class CaptionDecoderConfig:
    # llm_name: str = "Qwen/Qwen2-1.5B-Instruct"
    llm_name: str = "Qwen/Qwen2-0.5B-Instruct"
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    max_new_tokens: int = 64
    vl_projector_hidden: int = 2048
    vl_projector_layers: int = 2
    # llm_dim: int = 1536
    llm_dim: int = 896      # 从1536改成896，0.5B的hidden_dim是896
    freeze_except_lora: bool = True
    prompt_template: str = (
        "You are analyzing a pair of bi-temporal UAV remote sensing images captured "
        "over an urban construction site. The visual tokens encode change-aware "
        "spatial features derived from before and after imagery, supplemented by "
        "detection-guided spatial priors indicating changed regions. Describe the "
        "observed scene changes concisely and accurately.\nChanges: "
    )


@dataclass
class CLIPTextEncoderConfig:
    backbone: str = "openai/clip-vit-large-patch14"
    frozen: bool = True
    output_dim: int = 768


@dataclass
class LossConfig:
    lambda_caption: float = 1.0
    lambda_detection: float = 1.0
    lambda_align: float = 0.3
    dynamic_weight_temp: float = 2.0
    bce_pos_weight: float = 2.0
    infonce_temperature: float = 0.07


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-4
    vision_encoder_lr: float = 1e-5
    llm_lr: float = 1e-5
    weight_decay: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 2


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    warmup_lr_init: float = 1e-6


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 16
    val_batch_size: int = 4
    seed: int = 42
    amp: bool = False
    compile: bool = False
    ema: bool = False
    ema_decay: float = 0.999
    eval_freq: int = 1
    save_freq: int = 1
    log_freq: int = 10
    output_dir: str = "./outputs/ptnet_uccd"
    resume: Optional[str] = None
    pretrained: Optional[str] = None


@dataclass
class LogConfig:
    use_wandb: bool = True
    wandb_project: str = "PTNet"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    use_tensorboard: bool = True
    tensorboard_dir: str = "./runs"


@dataclass
class DistConfig:
    backend: str = "nccl"
    world_size: int = 2
    dist_url: str = "env://"
    find_unused_parameters: bool = False


@dataclass
class PTNetConfig:
    dataset: str = "uccd"
    experiment_name: str = "ptnet_uccd_baseline"

    data: DataConfig = field(default_factory=DataConfig)
    vision_encoder: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    prototype: PrototypeConfig = field(default_factory=PrototypeConfig)
    pg_cai: PGCAIConfig = field(default_factory=PGCAIConfig)
    tamg: TAMGConfig = field(default_factory=TAMGConfig)
    fpn: FPNConfig = field(default_factory=FPNConfig)
    mask_encoder: MaskEncoderConfig = field(default_factory=MaskEncoderConfig)
    caption_decoder: CaptionDecoderConfig = field(default_factory=CaptionDecoderConfig)
    clip_text: CLIPTextEncoderConfig = field(default_factory=CLIPTextEncoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)
    dist: DistConfig = field(default_factory=DistConfig)

    def __post_init__(self):
        assert self.data.img_size % self.vision_encoder.patch_size == 0, (
            f"img_size {self.data.img_size} must be divisible by "
            f"patch_size {self.vision_encoder.patch_size}"
        )
        n = (self.data.img_size // self.vision_encoder.patch_size) ** 2
        self.prototype.spatial_tokens = n
        Path(self.train.output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.prototype.save_path)).mkdir(parents=True, exist_ok=True)


def get_config(dataset: str = "uccd") -> PTNetConfig:
    cfg = PTNetConfig()

    if dataset == "whu_cdc":
        cfg.dataset = "whu_cdc"
        cfg.experiment_name = "ptnet_whu_cdc_baseline"
        cfg.prototype.num_clusters = 5
        cfg.prototype.save_path = "./cache/prototypes_whu_cdc.pt"
        cfg.data.root = "/path/to/whu_cdc"
        cfg.data.json_path = "/path/to/whu_cdc/annotations.json"
        cfg.train.output_dir = "./outputs/ptnet_whu_cdc"

    cfg.__post_init__()
    return cfg


UCCD_CONFIG = get_config("uccd")
WHU_CDC_CONFIG = get_config("whu_cdc")
