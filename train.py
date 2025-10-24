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
import random
import json
try:
    # Prefer new torch.amp API
    from torch.amp import autocast as torch_autocast, GradScaler as TorchGradScaler
except Exception:
    # Fallback for older PyTorch
    from torch.cuda.amp import autocast as torch_autocast, GradScaler as TorchGradScaler

from load_palette import load_class_palette
from datasets.graph_superpixel_dataset import GraphSuperpixelDataset
from models.gat import SPNodeRegressor
from torchvision.models import vgg16, VGG16_Weights
from utils.logger import TrainLogger
from utils.metrics import (
    collect_image_scores,
    calibrate_thresholds,
    example_based_metrics,
    collect_regression_data,
    mae_weighted_nodes,
    rmse_weighted_nodes,
    per_class_regression_scores_images,
    soft_dice_per_class,
    js_divergence_images,
    emd_1d_images,
)

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
    k_values: Sequence[int] = (60, 80)  # paper suggests ~50-90 superpixels
    cache_features: bool = True
    cache_dir: str = "artifacts/features"
    normalize_targets: bool = True
    feature_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precompute: bool = True
    hsv_threshold: float = 0.2
    feature_batch_size: int = 8
    precompute_workers: int = 0
    slic_backend: str = "cpu"

    # Loader
    batch_size: int = 1  # each graph is a sample; batching graphs of variable N kept as 1 for simplicity
    num_workers: int = 0
    shuffle: bool = True
    prefetch_factor: int = 2

    # Split (deterministic train/val within train dir)
    val_fraction: float = 0.1
    split_seed: int = 42

    # Model
    in_dim: int = 1024
    hidden_dim: int = 512
    out_activation: str = "none"  # raw logits for regression loss selection
    num_layers: int = 2
    num_heads: int = 3
    gat_dropout: float = 0.2
    integrate_dropout: float = 0.2
    normalize_node_features: bool = True

    # Training stability
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    loss_type: str = "kl"  # mse | kl (paper uses soft targets + KL)
    use_wandb: bool = False
    eval_every_epochs: int = 5
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    compile_fullgraph: bool = False

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
    """
    KL(target || softmax(logits)) averaged over VALID nodes only.

    - Masks out superpixels with zero mass after excluding the unknown class
      (i.e., rows where sum(target_probs)==0), since they carry no training signal.
    - Renormalizes valid target rows to sum exactly to 1 to ensure a proper distribution.
    """
    with torch.no_grad():
        row_sum = target_probs.sum(dim=-1, keepdim=True)
        valid_mask = (row_sum.squeeze(-1) > 0)

    if not torch.any(valid_mask):
        return pred_logits.sum() * 0.0  # zero loss if no valid nodes

    # Select only valid rows and renormalize to form exact distributions
    target_valid = target_probs[valid_mask]
    row_sum_valid = row_sum[valid_mask]
    target_valid = target_valid / row_sum_valid

    log_pred_valid = F.log_softmax(pred_logits[valid_mask], dim=-1)
    return F.kl_div(log_pred_valid, target_valid, reduction="batchmean")


def train_loop(cfg: TrainConfig) -> None:
    # Backend tuning
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    # Warm up VGG16 weights cache in the main process before DataLoader workers spawn
    try:
        _ = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    except Exception:
        pass

    names, class_rgb_values, unknown_index = load_class_palette(cfg.class_csv)
    img_size = None
    if cfg.img_size_w is not None and cfg.img_size_h is not None:
        img_size = (int(cfg.img_size_w), int(cfg.img_size_h))

    # Build and share a single VGG16 features backbone instance
    backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(cfg.feature_device).eval()

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
        hsv_threshold=cfg.hsv_threshold,
        normalize_targets=cfg.normalize_targets,
        precompute=cfg.precompute,
        backbone=backbone,
        feature_batch_size=cfg.feature_batch_size,
        precompute_workers=cfg.precompute_workers,
        slic_backend=cfg.slic_backend,
    )

    # Deterministic split of images into train/val (val excluded from training)
    num_images = len(ds.base_ds)
    image_indices = list(range(num_images))
    rng = random.Random(cfg.split_seed)
    rng.shuffle(image_indices)
    n_val = int(round(max(0.0, min(1.0, cfg.val_fraction)) * num_images))
    val_image_set = set(image_indices[:n_val]) if n_val > 0 else set()

    # Map to dataset sample indices (image_idx, k)
    train_sample_indices = [i for i, (img_idx, _k) in enumerate(ds.index_map) if img_idx not in val_image_set]
    # Note: we do not use the val subset here; kept only for evaluation to avoid leakage
    ds_train = torch.utils.data.Subset(ds, train_sample_indices)

    # Determine output dimension after excluding unknown
    num_classes_eff = len(class_rgb_values) - (1 if (unknown_index is not None and 0 <= unknown_index < len(class_rgb_values)) else 0)

    pin_memory = (torch.device(cfg.device).type == "cuda")
    loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_graphs,
        pin_memory=pin_memory,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
        pin_memory_device=(device.type if hasattr(torch.utils.data, 'DataLoader') else None),
    )

    device = torch.device(cfg.device)
    final_act = (None if cfg.out_activation in ("none", None) else cfg.out_activation)
    model = SPNodeRegressor(
        in_dim=cfg.in_dim,
        hidden_dim=cfg.hidden_dim,
        out_dim=num_classes_eff,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.gat_dropout,
        integrate_dropout=cfg.integrate_dropout,
        final_activation=final_act,
    ).to(device)

    # Optional compile
    try:
        if cfg.compile_model and hasattr(torch, "compile"):
            model = torch.compile(model, mode=cfg.compile_mode, fullgraph=bool(cfg.compile_fullgraph))
    except Exception:
        pass

    # Use fused AdamW when available on CUDA
    try:
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=(device.type == "cuda"))
    except TypeError:
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = TorchGradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build validation dataset with non-normalized targets for metrics
    ds_full_counts = GraphSuperpixelDataset(
        data_dir=cfg.data_dir,
        class_rgb_values=class_rgb_values,
        unknown_index=unknown_index,
        k_values=cfg.k_values,
        img_size=img_size,
        device=cfg.device,
        feature_device=cfg.feature_device,
        cache_features=cfg.cache_features,
        cache_dir=cfg.cache_dir,
        hsv_threshold=cfg.hsv_threshold,
        normalize_targets=False,
        precompute=False,  # avoid duplicate precompute; caches are already loaded by the first dataset
        backbone=backbone,
        feature_batch_size=cfg.feature_batch_size,
        precompute_workers=cfg.precompute_workers,
        slic_backend=cfg.slic_backend,
    )
    val_sample_indices = [i for i, (img_idx, _k) in enumerate(ds_full_counts.index_map) if img_idx in val_image_set]
    ds_val_counts = torch.utils.data.Subset(ds_full_counts, val_sample_indices)
    val_loader = DataLoader(
        ds_val_counts,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_graphs,
        pin_memory=pin_memory,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
        pin_memory_device=(device.type if hasattr(torch.utils.data, 'DataLoader') else None),
    )

    # Initialize logger
    logger = TrainLogger(use_wandb=cfg.use_wandb, project="graph-ml", run_name=None, config=cfg.__dict__)

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

            # Overlap H2D of next batch with compute of current batch when no AMP scaler sync
            if hasattr(loader, "_iterator") and hasattr(torch.cuda, "synchronize") and device.type == "cuda":
                torch.cuda.synchronize(device=None)

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
            # Step-wise logging
            if cfg.use_wandb:
                logger.log({"train/loss": float(loss.item()), "train/epoch": epoch}, step=step)
            step += 1

        # Save checkpoint after each epoch
        ckpt_path = out_dir / f"gat_regressor_epoch{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg.__dict__,
        }, ckpt_path)
        # Save sidecar JSON with architecture to ensure correct loading later
        arch = {
            "model_class": "SPNodeRegressor",
            "in_dim": cfg.in_dim,
            "hidden_dim": cfg.hidden_dim,
            "out_dim": num_classes_eff,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "dropout": cfg.gat_dropout,
            "integrate_dropout": cfg.integrate_dropout,
            "activation": "relu",
            "final_activation": final_act,
            "add_self_loops": False,
            "normalize_node_features": cfg.normalize_node_features,
            "split_seed": cfg.split_seed,
            "val_fraction": cfg.val_fraction,
        }
        # Persist the actual validation image ids (stems) for exact reproducibility
        try:
            val_ids = [Path(ds.base_ds.image_paths[i]).stem for i in sorted(val_image_set)]
        except Exception:
            val_ids = []
        meta = {"architecture": arch, "train_config": cfg.__dict__, "val_image_ids": val_ids}
        with open(ckpt_path.with_suffix('.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Saved checkpoint: {ckpt_path}")

        # Epoch-level logging
        if cfg.use_wandb:
            avg_loss = running / max(1, (i + 1))
            logger.log({"train/loss_epoch": float(avg_loss), "epoch": epoch}, step=step)

        # Periodic evaluation on validation split
        if (cfg.eval_every_epochs is not None) and (cfg.eval_every_epochs > 0) and ((epoch + 1) % int(cfg.eval_every_epochs) == 0):
            model.eval()
            with torch.inference_mode():
                # Classification-style metrics
                y_score_val, y_true_val = collect_image_scores(model, val_loader, cfg.normalize_node_features)
                thresholds = calibrate_thresholds(y_score_val, y_true_val, beta=2.0)
                p, r, f1, f2 = example_based_metrics(y_true_val, y_score_val, thresholds)

                # Regression-style metrics
                reg = collect_regression_data(model, val_loader, cfg.normalize_node_features)
                metrics_to_log = {
                    "val/precision": float(p),
                    "val/recall": float(r),
                    "val/f1": float(f1),
                    "val/f2": float(f2),
                }
                nodes = reg["nodes"]
                if nodes["y_true"].numel() > 0:
                    y_true_nodes = nodes["y_true"].numpy()
                    y_pred_nodes = nodes["y_pred"].numpy()
                    w_nodes = nodes["weights"].numpy()
                    metrics_to_log.update({
                        "val/node_mae": float(mae_weighted_nodes(y_true_nodes, y_pred_nodes, w_nodes)),
                        "val/node_rmse": float(rmse_weighted_nodes(y_true_nodes, y_pred_nodes, w_nodes)),
                    })
                images = reg["images"]
                if images["y_true"].numel() > 0:
                    y_true_img = images["y_true"].numpy()
                    y_pred_img = images["y_pred"].numpy()
                    w_img = images["weights"].numpy()
                    img_scores = per_class_regression_scores_images(y_true_img, y_pred_img, w_img)
                    dice_pc, dice_macro = soft_dice_per_class(y_true_img, y_pred_img)
                    js_img = js_divergence_images(y_true_img, y_pred_img)
                    emd_img = emd_1d_images(y_true_img, y_pred_img)
                    metrics_to_log.update({
                        "val/img_mae_macro": float(img_scores["macro"]["mae"]),
                        "val/img_rmse_macro": float(img_scores["macro"]["rmse"]),
                        "val/img_r2_macro": float(img_scores["macro"]["r2"]),
                        "val/img_smape_macro": float(img_scores["macro"]["smape"]),
                        "val/img_js": float(js_img),
                        "val/img_dice_macro": float(dice_macro),
                        "val/img_emd": float(emd_img),
                    })

                # Print concise summary
                print(f"Eval@epoch {epoch}: F1={metrics_to_log.get('val/f1', float('nan')):.4f} F2={metrics_to_log.get('val/f2', float('nan')):.4f} "
                      f"nodeMAE={metrics_to_log.get('val/node_mae', float('nan')):.4f} imgMAE={metrics_to_log.get('val/img_mae_macro', float('nan')):.4f}")

                if cfg.use_wandb:
                    logger.log(metrics_to_log, step=step)

    # Finish logger
    try:
        logger.finish()
    except Exception:
        pass


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
    p.add_argument("--hsv_threshold", type=float, default=None)
    p.add_argument("--feature_batch_size", type=int, default=None)
    p.add_argument("--precompute_workers", type=int, default=None)
    p.add_argument("--slic_backend", type=str, default=None, choices=["cpu"])  # fixed to skimage

    # Loader
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--no_shuffle", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=None)

    # Model
    p.add_argument("--in_dim", type=int, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--out_activation", type=str, default=None, choices=["relu", "sigmoid", "none"])
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    p.add_argument("--gat_dropout", type=float, default=None)
    p.add_argument("--integrate_dropout", type=float, default=None)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--eval_every_epochs", type=int, default=None)
    p.add_argument("--compile_model", action="store_true")
    p.add_argument("--compile_mode", type=str, default=None)
    p.add_argument("--compile_fullgraph", action="store_true")

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
    if hasattr(args, "use_wandb") and args.use_wandb:
        cfg_dict["use_wandb"] = True
    if hasattr(args, "eval_every_epochs") and (args.eval_every_epochs is not None):
        cfg_dict["eval_every_epochs"] = int(args.eval_every_epochs)
    if hasattr(args, "feature_batch_size") and (args.feature_batch_size is not None):
        cfg_dict["feature_batch_size"] = int(args.feature_batch_size)
    if hasattr(args, "precompute_workers") and (args.precompute_workers is not None):
        cfg_dict["precompute_workers"] = int(args.precompute_workers)
    if hasattr(args, "slic_backend") and (args.slic_backend is not None):
        cfg_dict["slic_backend"] = str(args.slic_backend)
    if hasattr(args, "prefetch_factor") and (args.prefetch_factor is not None):
        cfg_dict["prefetch_factor"] = int(args.prefetch_factor)
    if hasattr(args, "compile_model") and args.compile_model:
        cfg_dict["compile_model"] = True
    if hasattr(args, "compile_mode") and (args.compile_mode is not None):
        cfg_dict["compile_mode"] = str(args.compile_mode)
    if hasattr(args, "compile_fullgraph") and args.compile_fullgraph:
        cfg_dict["compile_fullgraph"] = True

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


