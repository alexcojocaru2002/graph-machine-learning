from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
try:
    # Prefer new torch.amp API
    from torch.amp import autocast as torch_autocast, GradScaler as TorchGradScaler
except Exception:
    # Fallback for older PyTorch
    from torch.cuda.amp import autocast as torch_autocast, GradScaler as TorchGradScaler

from load_palette import load_class_palette
from datasets.graph_superpixel_dataset import GraphSuperpixelDataset
from models.gat import SPNodeRegressor

try:
    import yaml
except Exception:
    yaml = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data/train"
    class_csv: str = "data/class_dict.csv"
    img_size_w: Optional[int] = None
    img_size_h: Optional[int] = None
    k_values: Sequence[int] = (400, 600)
    cache_features: bool = True
    cache_dir: str = "artifacts/features"
    normalize_targets: bool = True
    feature_device: str = "cpu"  # device for CNN feature extraction (cpu recommended)
    precompute: bool = True

    # Loader
    batch_size: int = 1  # each graph is a sample; batching graphs of variable N kept as 1 for simplicity
    num_workers: int = 0
    shuffle: bool = True

    # Model
    in_dim: int = 2048
    hidden_dim: int = 256
    out_activation: str = "relu"  # positive areas
    num_layers: int = 3
    num_heads: int = 4
    gat_dropout: float = 0.2
    integrate_dropout: float = 0.2
    normalize_node_features: bool = True

    # Training stability
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    loss_type: str = "mse"  # mse | kl

    # Optim
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 20

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging / ckpt
    out_dir: str = "artifacts/"  # save model checkpoints
    log_interval: int = 10


def collate_graphs(batch):
    """
    Collate a list of variable-sized graphs into one batched graph by concatenating nodes
    and offsetting edge indices. Returns dict with x [sumN,F], edge_index [2,sumE], y [sumN,C].
    """
    if len(batch) == 1:
        b = batch[0]
        return {"x": b["x"], "edge_index": b["edge_index"], "y": b["y"], "meta": {"num_graphs": 1, "nodes_per_graph": [b["x"].shape[0]]}}

    nodes_per_graph = [sample["x"].shape[0] for sample in batch]
    offsets = []
    total = 0
    for n in nodes_per_graph:
        offsets.append(total)
        total += int(n)

    x_cat = torch.cat([sample["x"] for sample in batch], dim=0)
    y_cat = torch.cat([sample["y"] for sample in batch], dim=0)

    edge_list = []
    for offset, sample in zip(offsets, batch):
        ei = sample["edge_index"] + offset
        edge_list.append(ei)
    edge_index_cat = torch.cat(edge_list, dim=1)

    meta = {"num_graphs": len(batch), "nodes_per_graph": nodes_per_graph}
    return {"x": x_cat, "edge_index": edge_index_cat, "y": y_cat, "meta": meta}


def mse_loss_area(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def kl_loss_area(pred_logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    # Expect target_probs to be probabilities; small epsilon for numerical stability
    eps = 1e-8
    target = target_probs.clamp(min=eps)
    log_pred = F.log_softmax(pred_logits, dim=-1)
    return F.kl_div(log_pred, target, reduction="batchmean")


def train_loop(cfg: TrainConfig) -> None:
    names, class_rgb_values, unknown_index = load_class_palette(cfg.class_csv)
    img_size = None
    if cfg.img_size_w is not None and cfg.img_size_h is not None:
        img_size = (int(cfg.img_size_w), int(cfg.img_size_h))

    ds = GraphSuperpixelDataset(
        data_dir=cfg.data_dir,
        class_rgb_values=class_rgb_values,
        unknown_index=unknown_index,
        k_values=cfg.k_values,
        img_size=img_size,
        device=cfg.device,
        feature_device=cfg.feature_device,
        cache_features=cfg.cache_features,
        cache_dir=cfg.cache_dir,
        normalize_targets=cfg.normalize_targets,
        precompute=cfg.precompute,
    )

    # Determine output dimension after excluding unknown
    num_classes_eff = len(class_rgb_values) - (1 if (unknown_index is not None and 0 <= unknown_index < len(class_rgb_values)) else 0)

    pin_memory = (torch.device(cfg.device).type == "cuda")
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_graphs,
        pin_memory=pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )

    device = torch.device(cfg.device)
    model = SPNodeRegressor(
        in_dim=cfg.in_dim,
        hidden_dim=cfg.hidden_dim,
        out_dim=num_classes_eff,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.gat_dropout,
        integrate_dropout=cfg.integrate_dropout,
        final_activation=cfg.out_activation,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = TorchGradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        iterator = loader
        if tqdm is not None:
            iterator = tqdm(loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(iterator):
            optimizer.zero_grad(set_to_none=True)

            x = batch["x"].to(device, non_blocking=True)
            if cfg.normalize_node_features:
                x = F.normalize(x, p=2, dim=1)
            edge_index = batch["edge_index"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            with torch_autocast(device_type=device.type, enabled=(cfg.use_amp and device.type == "cuda")):
                pred = model(x, edge_index)
                if cfg.loss_type == "mse":
                    loss = mse_loss_area(pred, y)
                elif cfg.loss_type == "kl":
                    loss = kl_loss_area(pred, y)
                else:
                    raise ValueError(f"Unknown loss_type: {cfg.loss_type}")

            scaler.scale(loss).backward()

            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running += float(loss.item())
            if tqdm is not None:
                avg = running / (i + 1)
                iterator.set_postfix({"loss": f"{avg:.6f}"})
            if (step % cfg.log_interval) == 0 and tqdm is None:
                print(f"epoch {epoch} step {step}: loss {running / (i + 1):.6f}")
            step += 1

        # Save checkpoint after each epoch
        ckpt_path = out_dir / f"gat_regressor_epoch{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg.__dict__,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GAT-based superpixel area regressor")
    p.add_argument("--config", type=str, default=None, help="YAML config file path")

    # Data
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--class_csv", type=str, default=None)
    p.add_argument("--img_size_w", type=int, default=None)
    p.add_argument("--img_size_h", type=int, default=None)
    p.add_argument("--k_values", type=int, nargs="+", default=None)
    p.add_argument("--cache_features", action="store_true")
    p.add_argument("--no_cache_features", action="store_true")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--normalize_targets", action="store_true")
    p.add_argument("--no_normalize_targets", action="store_true")
    p.add_argument("--feature_device", type=str, default=None)

    # Loader
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--no_shuffle", action="store_true")

    # Model
    p.add_argument("--in_dim", type=int, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--out_activation", type=str, default=None, choices=["relu", "sigmoid", "none"])
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    p.add_argument("--gat_dropout", type=float, default=None)
    p.add_argument("--integrate_dropout", type=float, default=None)

    # Optim
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)

    # Device
    p.add_argument("--device", type=str, default=None)

    # Logging
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--log_interval", type=int, default=None)

    return p.parse_args()


def load_config_yaml(path: str | None) -> dict:
    if path is None:
        return {}
    if yaml is None:
        raise RuntimeError("pyyaml is not installed but a config file was provided")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def merge_config(default: TrainConfig, yaml_cfg: dict, args: argparse.Namespace) -> TrainConfig:
    cfg_dict = default.__dict__.copy()
    cfg_dict.update({k: v for k, v in yaml_cfg.items() if v is not None})

    # Map CLI overrides
    for key in cfg_dict.keys():
        if hasattr(args, key):
            val = getattr(args, key)
            if val is not None:
                cfg_dict[key] = val

    # Booleans with dual flags
    if args.cache_features:
        cfg_dict["cache_features"] = True
    if args.no_cache_features:
        cfg_dict["cache_features"] = False
    if args.normalize_targets:
        cfg_dict["normalize_targets"] = True
    if args.no_normalize_targets:
        cfg_dict["normalize_targets"] = False
    if args.no_shuffle:
        cfg_dict["shuffle"] = False

    # k_values may be provided as space-separated ints
    kv = cfg_dict.get("k_values")
    if isinstance(kv, (list, tuple)):
        cfg_dict["k_values"] = [int(x) for x in kv]

    # img_size handling (keep as w,h in cfg; dataset will build tuple)
    return TrainConfig(**cfg_dict)


def main():
    args = parse_args()
    base = TrainConfig()
    yaml_cfg = load_config_yaml(args.config)
    cfg = merge_config(base, yaml_cfg, args)
    print("Resolved config:\n", cfg)
    train_loop(cfg)


if __name__ == "__main__":
    main()


