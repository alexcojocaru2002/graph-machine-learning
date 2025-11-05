from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class EvalConfig:
    # Data (explicit train/valid/test roots)
    train_dir: str = "../data/train"
    valid_dir: str = "../data/valid"
    test_dir: str = "../data/test"
    class_csv: str = "data/class_dict.csv"
    img_size_w: Optional[int] = 512
    img_size_h: Optional[int] = 512
    k_values: Sequence[int] = (60,)
    cache_dir: str = "artifacts/features"
    feature_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hsv_threshold: float = 0.2
    feature_batch_size: int = 8
    slic_backend: str = "cpu"

    # Loader
    batch_size: int = 16
    num_workers: int = 2
    prefetch_factor: int = 2

    # Model
    in_dim: int = 1024
    hidden_dim: int = 512
    num_layers: int = 2
    num_heads: int = 3
    gat_dropout: float = 0.2
    integrate_dropout: float = 0.2
    normalize_node_features: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Logging
    use_wandb: bool = False
    wandb_project: str = "graph-ml"

    # Checkpoint
    ckpt_path: Optional[str] = None
    ckpt2_path: Optional[str] = None

    # Plotting
    plot_sample: bool = False
    plot_k: Optional[int] = None
    save_plots_dir: str = "../artifacts/plots"
    no_show: bool = True
    plot_image: Optional[str] = None  # image stem or index to use for plotting

