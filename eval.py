from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import vgg16, VGG16_Weights
import os
import random
import matplotlib
import matplotlib.pyplot as plt

from load_palette import load_class_palette
from datasets.graph_superpixel_dataset import GraphSuperpixelDataset
from models.gat import SPNodeRegressor
from train import collate_graphs as train_collate_graphs
from utils.logger import TrainLogger
import json
try:
    from torch.amp import autocast as torch_autocast
except Exception:
    from torch.cuda.amp import autocast as torch_autocast


@dataclass
class EvalConfig:
    # Data (explicit train/valid/test roots)
    train_dir: str = "data/train"
    valid_dir: str = "data/valid"
    test_dir: str = "data/test"
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
    save_plots_dir: str = "artifacts/plots"
    no_show: bool = True
    plot_image: Optional[str] = None  # image stem or index to use for plotting


def collate_graphs(batch):
    return train_collate_graphs(batch)


def make_dataset(root: str, class_csv: str, img_size: Optional[Tuple[int, int]], k_values: Sequence[int], feature_device: str, cache_dir: str, normalize_targets: bool, backbone: torch.nn.Module, device: Optional[str] = None, hsv_threshold: float = 0.2) -> Tuple[GraphSuperpixelDataset, List[str], List[Tuple[int, int, int]], Optional[int]]:
    names, class_rgb_values, unknown_index = load_class_palette(class_csv)
    ds = GraphSuperpixelDataset(
        data_dir=root,
        class_rgb_values=class_rgb_values,
        unknown_index=unknown_index,
        k_values=k_values,
        img_size=img_size,
        device=(device or ("cuda" if torch.cuda.is_available() else "cpu")),
        feature_device=feature_device,
        cache_features=True,
        cache_dir=cache_dir,
        normalize_targets=normalize_targets,
        precompute=False,
        backbone=backbone,
        hsv_threshold=hsv_threshold,
        feature_batch_size=8,
        slic_backend="auto",
    )
    return ds, names, class_rgb_values, unknown_index


# -----------------------------
# Classification-oriented helpers (kept)
# -----------------------------
def image_scores_from_nodes(logits: torch.Tensor, y_counts: torch.Tensor) -> torch.Tensor:
    # logits: [N,C], y_counts: [N,C] counts per class excluding unknown
    prob = F.softmax(logits, dim=-1)
    weights = y_counts.sum(dim=-1)  # [N]
    weights = torch.where(weights > 0, weights, torch.ones_like(weights))
    s = (prob * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()
    return s  # [C]


def image_level_ground_truth(mask_counts: torch.Tensor) -> torch.Tensor:
    # mask_counts: [N,C] counts per class per superpixel (unknown excluded)
    per_class_total = mask_counts.sum(dim=0)
    return (per_class_total > 0).float()


def collect_image_scores(model: SPNodeRegressor, loader: DataLoader, normalize_node_features: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    y_scores = []
    y_true = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            if normalize_node_features:
                x = F.normalize(x, p=2, dim=1)
            edge_index = batch["edge_index"].to(device, non_blocking=True)
            # Need counts per class for weights: build a non-normalized dataset for evaluation
            # Here, loader must have been constructed with normalize_targets=False
            y_counts = batch["y"].to(device, non_blocking=True)
            with torch_autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(x, edge_index)
            # Split per-graph using nodes_per_graph metadata
            meta = batch.get("meta", {}) if isinstance(batch, dict) else {}
            npg = meta.get("nodes_per_graph", None)
            if not isinstance(npg, list) or len(npg) == 0:
                npg = [x.shape[0]]
            offset = 0
            for n in npg:
                n = int(n)
                logits_g = logits[offset:offset+n]
                y_counts_g = y_counts[offset:offset+n]
                s = image_scores_from_nodes(logits_g, y_counts_g)
                y_scores.append(s.cpu())
                y_true.append(image_level_ground_truth(y_counts_g).cpu())
                offset += n
    return torch.stack(y_scores, dim=0), torch.stack(y_true, dim=0)


def calibrate_thresholds(y_score: torch.Tensor, y_true: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    C = y_score.shape[1]
    thresholds = torch.zeros(C)
    for c in range(C):
        scores = y_score[:, c]
        labels = y_true[:, c]
        best_t = 0.5
        best_fbeta = -1.0
        # Sweep thresholds
        for t in torch.linspace(0.01, 0.99, steps=99):
            pred = (scores >= t).float()
            tp = (pred * labels).sum()
            fp = (pred * (1 - labels)).sum()
            fn = ((1 - pred) * labels).sum()
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            fbeta = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall + 1e-9)
            val = fbeta.item()
            if val > best_fbeta:
                best_fbeta = val
                best_t = float(t.item())
        thresholds[c] = best_t
    return thresholds


def example_based_metrics(y_true_bin: torch.Tensor, y_score: torch.Tensor, thresholds: torch.Tensor) -> Tuple[float, float, float, float]:
    y_pred = (y_score >= thresholds.unsqueeze(0)).float()
    tp = (y_pred * y_true_bin).sum(dim=1)
    fp = (y_pred * (1 - y_true_bin)).sum(dim=1)
    fn = ((1 - y_pred) * y_true_bin).sum(dim=1)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = (2 * precision * recall) / (precision + recall + 1e-9)
    f2 = (5 * precision * recall) / (4 * precision + recall + 1e-9)
    return precision.mean().item(), recall.mean().item(), f1.mean().item(), f2.mean().item()


# -----------------------------
# Regression-oriented helpers and metrics
# -----------------------------
def _safe_row_normalize(arr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    row_sum = arr.sum(dim=-1, keepdim=True)
    valid = (row_sum.squeeze(-1) > 0)
    row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
    return arr / row_sum, valid


@torch.inference_mode()
def collect_regression_data(model: SPNodeRegressor, loader: DataLoader, normalize_node_features: bool) -> dict:
    device = next(model.parameters()).device
    model.eval()
    pred_nodes: List[torch.Tensor] = []
    true_nodes: List[torch.Tensor] = []
    node_weights: List[torch.Tensor] = []

    pred_images: List[torch.Tensor] = []
    true_images: List[torch.Tensor] = []
    image_weights: List[torch.Tensor] = []

    with torch.inference_mode():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            if normalize_node_features:
                x = F.normalize(x, p=2, dim=1)
            edge_index = batch["edge_index"].to(device, non_blocking=True)
            y_counts = batch["y"].to(device, non_blocking=True)  # [N,C] counts per node
            logits = model(x, edge_index)
            prob = F.softmax(logits, dim=-1)  # [N,C]

            # Split per-graph
            meta = batch.get("meta", {}) if isinstance(batch, dict) else {}
            npg = meta.get("nodes_per_graph", None)
            if not isinstance(npg, list) or len(npg) == 0:
                npg = [x.shape[0]]
            offset = 0
            for n in npg:
                n = int(n)
                prob_g = prob[offset:offset+n]
                counts_g = y_counts[offset:offset+n]

                # Node-level true fractions and weights
                true_frac_g, valid_mask_g = _safe_row_normalize(counts_g)
                weights_g = counts_g.sum(dim=-1)  # pixels per node

                # Keep only valid nodes
                if valid_mask_g.any():
                    pred_nodes.append(prob_g[valid_mask_g].detach().cpu())
                    true_nodes.append(true_frac_g[valid_mask_g].detach().cpu())
                    node_weights.append(weights_g[valid_mask_g].detach().cpu())

                # Image-level aggregated distributions
                total_pixels = weights_g.sum()
                if float(total_pixels.item()) > 0:
                    pred_counts_img = (prob_g * weights_g.unsqueeze(-1)).sum(dim=0)  # [C]
                    true_counts_img = counts_g.sum(dim=0)  # [C]
                    pred_img = pred_counts_img / total_pixels
                    true_img = true_counts_img / total_pixels
                    pred_images.append(pred_img.detach().cpu())
                    true_images.append(true_img.detach().cpu())
                    image_weights.append(total_pixels.detach().cpu())
                offset += n

    out = {
        "nodes": {
            "y_pred": (torch.cat(pred_nodes, dim=0) if len(pred_nodes) else torch.empty(0)),
            "y_true": (torch.cat(true_nodes, dim=0) if len(true_nodes) else torch.empty(0)),
            "weights": (torch.cat(node_weights, dim=0) if len(node_weights) else torch.empty(0)),
        },
        "images": {
            "y_pred": (torch.stack(pred_images, dim=0) if len(pred_images) else torch.empty(0)),
            "y_true": (torch.stack(true_images, dim=0) if len(true_images) else torch.empty(0)),
            "weights": (torch.stack(image_weights, dim=0) if len(image_weights) else torch.empty(0)),
        }
    }
    return out


def _weighted_mean(values: np.ndarray, weights: Optional[np.ndarray]) -> float:
    if weights is None:
        return float(np.mean(values))
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim > 1:
        w = w.reshape(-1)
    return float(np.average(values, weights=w))


def mae_weighted_nodes(y_true: np.ndarray, y_pred: np.ndarray, node_weights: np.ndarray) -> float:
    # per-node mean absolute error across classes, then weighted across nodes
    per_node = np.mean(np.abs(y_true - y_pred), axis=1)
    return _weighted_mean(per_node, node_weights)


def rmse_weighted_nodes(y_true: np.ndarray, y_pred: np.ndarray, node_weights: np.ndarray) -> float:
    per_node_mse = np.mean((y_true - y_pred) ** 2, axis=1)
    return float(np.sqrt(_weighted_mean(per_node_mse, node_weights)))


def brier_score_weighted_nodes(y_true: np.ndarray, y_pred: np.ndarray, node_weights: np.ndarray) -> float:
    per_node_mse = np.mean((y_true - y_pred) ** 2, axis=1)
    return _weighted_mean(per_node_mse, node_weights)


def _js_divergence_batch(p: np.ndarray, q: np.ndarray, weights: Optional[np.ndarray] = None, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=1)
    js = 0.5 * (kl_pm + kl_qm)  # [N]
    return _weighted_mean(js, weights)


def _hellinger_batch(p: np.ndarray, q: np.ndarray, weights: Optional[np.ndarray] = None, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    d = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis=1)) / np.sqrt(2.0)
    return _weighted_mean(d, weights)


def _cosine_sim_batch(p: np.ndarray, q: np.ndarray, weights: Optional[np.ndarray] = None, eps: float = 1e-12) -> float:
    num = np.sum(p * q, axis=1)
    den = (np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1) + eps)
    sim = num / den
    return _weighted_mean(sim, weights)


def _r2_weighted(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray]) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if sample_weight is None:
        y_bar = float(np.mean(y_true))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_bar) ** 2)
    else:
        w = np.asarray(sample_weight, dtype=np.float64)
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            return float("nan")
        y_bar = float(np.sum(w * y_true) / w_sum)
        ss_res = float(np.sum(w * (y_true - y_pred) ** 2))
        ss_tot = float(np.sum(w * (y_true - y_bar) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - (ss_res / (ss_tot + 1e-12))


def r2_per_class_nodes(y_true: np.ndarray, y_pred: np.ndarray, node_weights: np.ndarray) -> Tuple[dict, float]:
    C = y_true.shape[1]
    scores = {}
    vals = []
    for c in range(C):
        s = _r2_weighted(y_true[:, c], y_pred[:, c], node_weights)
        scores[c] = float(s)
        if not np.isnan(s):
            vals.append(float(s))
    macro = float(np.mean(vals)) if len(vals) else float("nan")
    return scores, macro


def _pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.astype(np.float64)
    yp = y_pred.astype(np.float64)
    yt = yt - yt.mean()
    yp = yp - yp.mean()
    denom = (np.linalg.norm(yt) * np.linalg.norm(yp))
    if denom == 0.0:
        return 0.0
    return float(np.dot(yt, yp) / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    # Average ranks for ties (like scipy.stats.rankdata with method='average')
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
    # handle ties
    sorted_a = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + 1 + j + 1)
            ranks[order[i:j+1]] = avg
        i = j + 1
    return ranks


def pearson_spearman_per_class_nodes(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[dict, dict, float, float]:
    C = y_true.shape[1]
    pearson = {}
    spearman = {}
    pvals = []
    svals = []
    for c in range(C):
        yt = y_true[:, c]
        yp = y_pred[:, c]
        pear = _pearsonr(yt, yp)
        pearson[c] = float(pear)
        pvals.append(pear)
        rt = _rankdata(yt)
        rp = _rankdata(yp)
        spear = _pearsonr(rt, rp)
        spearman[c] = float(spear)
        svals.append(spear)
    return pearson, spearman, float(np.mean(pvals)), float(np.mean(svals))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_true - y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = np.where(denom > 0, diff / denom, 0.0)
    return 100.0 * float(np.mean(frac))


def soft_dice_per_class(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> Tuple[dict, float]:
    # y_true, y_pred: [M,C] per-image fractions
    C = y_true.shape[1]
    scores = {}
    vals = []
    for c in range(C):
        t = y_true[:, c]
        p = y_pred[:, c]
        num = 2.0 * (t * p)
        den = (t + p) + eps
        s = float(np.mean(num / den))
        scores[c] = s
        vals.append(s)
    return scores, float(np.mean(vals))


def js_divergence_images(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # average per-image JS divergence between distributions
    vals = []
    for i in range(y_true.shape[0]):
        vals.append(_js_divergence_batch(y_true[i:i+1], y_pred[i:i+1]))
    return float(np.mean(vals)) if len(vals) else float("nan")


def emd_1d_images(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # 1D Earth Mover's Distance with uniform bin spacing using CDF difference
    def emd_1d(p: np.ndarray, q: np.ndarray) -> float:
        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)
        return float(np.sum(np.abs(cdf_p - cdf_q)))
    vals = [emd_1d(y_true[i], y_pred[i]) for i in range(y_true.shape[0])]
    return float(np.mean(vals)) if len(vals) else float("nan")


def per_class_regression_scores_images(y_true: np.ndarray, y_pred: np.ndarray, image_weights: Optional[np.ndarray] = None) -> dict:
    # compute per-class MAE, RMSE, R2, sMAPE with macro/micro averages
    C = y_true.shape[1]
    out = {"per_class": {}, "macro": {}, "micro": {}}
    mae_vals = []
    rmse_vals = []
    r2_vals = []
    smape_vals = []
    # micro accumulators
    mae_num = 0.0
    mae_den = 0.0
    rmse_num = 0.0
    rmse_den = 0.0
    for c in range(C):
        t = y_true[:, c]
        p = y_pred[:, c]
        w = image_weights
        err = np.abs(t - p)
        mae_c = _weighted_mean(err, w)
        rmse_c = float(np.sqrt(_weighted_mean((t - p) ** 2, w)))
        r2_c = _r2_weighted(t, p, w)
        smape_c = smape(t, p)
        out["per_class"][c] = {"mae": mae_c, "rmse": rmse_c, "r2": r2_c, "smape": smape_c}
        mae_vals.append(mae_c)
        rmse_vals.append(rmse_c)
        r2_vals.append(r2_c)
        smape_vals.append(smape_c)
        if w is None:
            mae_num += float(np.mean(err))
            mae_den += 1.0
            rmse_num += float(np.mean((t - p) ** 2))
            rmse_den += 1.0
        else:
            mae_num += float(np.sum(w * err) / (np.sum(w) + 1e-12))
            mae_den += 1.0
            rmse_num += float(np.sum(w * (t - p) ** 2) / (np.sum(w) + 1e-12))
            rmse_den += 1.0
    out["macro"] = {
        "mae": float(np.mean(mae_vals)),
        "rmse": float(np.mean(rmse_vals)),
        "r2": float(np.mean(r2_vals)),
        "smape": float(np.mean(smape_vals)),
    }
    out["micro"] = {
        "mae": float(mae_num / max(mae_den, 1.0)),
        "rmse": float(np.sqrt(rmse_num / max(rmse_den, 1.0))),
    }
    return out


def load_model(cfg: EvalConfig, num_classes_eff: int) -> SPNodeRegressor:
    device = torch.device(cfg.device)
    # Try to load sidecar JSON for architecture
    arch_path = Path(cfg.ckpt_path).with_suffix('.json') if cfg.ckpt_path else None
    arch = None
    if arch_path is not None and arch_path.exists():
        with open(arch_path, 'r') as f:
            meta = json.load(f)
        arch = meta.get("architecture", None)

    if arch is not None:
        # Build model from recorded architecture, with guard against class count mismatch
        out_dim = arch.get("out_dim", num_classes_eff)
        if out_dim != num_classes_eff:
            # Prefer dataset-driven class count to avoid silent mismatch
            out_dim = num_classes_eff
        model = SPNodeRegressor(
            in_dim=arch.get("in_dim", cfg.in_dim),
            hidden_dim=arch.get("hidden_dim", cfg.hidden_dim),
            out_dim=out_dim,
            num_layers=arch.get("num_layers", cfg.num_layers),
            num_heads=arch.get("num_heads", cfg.num_heads),
            dropout=arch.get("dropout", cfg.gat_dropout),
            integrate_dropout=arch.get("integrate_dropout", cfg.integrate_dropout),
            activation=arch.get("activation", "relu"),
            final_activation=arch.get("final_activation", None),
        ).to(device)
    else:
        # Fallback to CLI/config
        model = SPNodeRegressor(
            in_dim=cfg.in_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=num_classes_eff,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.gat_dropout,
            integrate_dropout=cfg.integrate_dropout,
            final_activation=None,
        ).to(device)

    if cfg.ckpt_path is not None:
        ckpt = torch.load(cfg.ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _mask_idx_to_rgb(mask_idx: np.ndarray, class_rgb_values: List[Tuple[int, int, int]]) -> np.ndarray:
    H, W = mask_idx.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, (r, g, b) in enumerate(class_rgb_values):
        out[mask_idx == idx] = (r, g, b)
    return out


def _effective_palette(class_rgb_values: List[Tuple[int, int, int]], unknown_index: Optional[int]) -> List[Tuple[int, int, int]]:
    if unknown_index is None or unknown_index < 0 or unknown_index >= len(class_rgb_values):
        return list(class_rgb_values)
    return [c for i, c in enumerate(class_rgb_values) if i != unknown_index]


def _plot_random_sample_for_k(
    ds_full: GraphSuperpixelDataset,
    model_list: List[SPNodeRegressor],
    model_labels: List[str],
    names: List[str],
    class_rgb_values: List[Tuple[int, int, int]],
    unknown_index: Optional[int],
    k_value: int,
    normalize_node_features: bool,
    device: torch.device,
    save_dir: str,
    show: bool,
    image_index: Optional[int] = None,
) -> None:
    # Choose a random image from the base dataset
    if image_index is None:
        rng = random.Random()
        img_idx = rng.randrange(len(ds_full.base_ds))
    else:
        img_idx = int(max(0, min(len(ds_full.base_ds) - 1, image_index)))
    # Find the linear index for (img_idx, k_value)
    sample_idx = None
    for i, (im_i, k) in enumerate(ds_full.index_map):
        if im_i == img_idx and int(k) == int(k_value):
            sample_idx = i
            break
    if sample_idx is None:
        print(f"[plot] Could not find sample for chosen image and k={k_value}")
        return

    sample = ds_full[sample_idx]
    x = sample["x"].to(device)
    if normalize_node_features:
        x = F.normalize(x, p=2, dim=1)
    edge_index = sample["edge_index"].to(device)

    # Load image and mask RGB
    img_t, img_rgb, mask_t = ds_full.base_ds[img_idx]
    mask_idx = mask_t.numpy().astype(np.int64)
    mask_rgb = _mask_idx_to_rgb(mask_idx, class_rgb_values)

    # Load superpixel map from cache (fast path)
    img_path = sample["meta"]["image_path"]
    k = int(sample["meta"]["k"])
    x_path, sp_path = ds_full._features_cache_paths_npy(img_path, k, ds_full.base_ds.img_size)  # type: ignore[attr-defined]
    if sp_path.exists():
        sp = np.load(sp_path, mmap_mode='r').astype(np.int64)
    else:
        # Fallback: recompute SLIC if cache missing
        from utils.graph_utils import slic_labels
        sp = slic_labels(img_rgb, n_segments=k, compactness=ds_full.slic_compactness, sigma=ds_full.slic_sigma, start_label=ds_full.slic_start_label)

    # Model predictions per node -> per-pixel map via superpixels
    eff_palette = _effective_palette(class_rgb_values, unknown_index)
    pred_rgbs: List[np.ndarray] = []
    for model in model_list:
        with torch.inference_mode():
            logits = model(x, edge_index)
            pred_cls = torch.argmax(logits, dim=-1).detach().cpu().numpy()  # [N]
        # Map node class to pixels
        H, W = sp.shape
        pred_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        # Note: pred class indices correspond to effective palette (unknown removed)
        for node_id in range(pred_cls.shape[0]):
            c = int(pred_cls[node_id])
            if c < 0 or c >= len(eff_palette):
                color = (0, 0, 0)
            else:
                color = eff_palette[c]
            pred_rgb[sp == node_id] = color
        pred_rgbs.append(pred_rgb)

    # Build figure
    ncols = 2 + len(pred_rgbs)  # image, mask, preds...
    plt.figure(figsize=(4 * ncols, 4))
    ax = plt.subplot(1, ncols, 1)
    ax.imshow(img_rgb)
    ax.set_title("Image")
    ax.axis('off')

    ax = plt.subplot(1, ncols, 2)
    ax.imshow(mask_rgb)
    ax.set_title("Mask")
    ax.axis('off')

    for j, pred_rgb in enumerate(pred_rgbs):
        ax = plt.subplot(1, ncols, 3 + j)
        ax.imshow(pred_rgb)
        ax.set_title(f"Pred: {model_labels[j]}")
        ax.axis('off')

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"sample_k{k}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def _grouped_bar(ax, labels: List[str], series: List[List[float]], series_labels: List[str]):
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(series))
    for i, vals in enumerate(series):
        ax.bar(x + i * width, vals, width, label=series_labels[i])
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend()


def _plot_superpixel_distribution_for_k(
    ds_full: GraphSuperpixelDataset,
    model_list: List[SPNodeRegressor],
    model_labels: List[str],
    class_rgb_values: List[Tuple[int, int, int]],
    unknown_index: Optional[int],
    k_value: int,
    normalize_node_features: bool,
    device: torch.device,
    save_dir: str,
    show: bool,
    image_index: Optional[int] = None,
) -> None:
    # Choose image index deterministically if provided
    if image_index is None:
        rng = random.Random()
        img_idx = rng.randrange(len(ds_full.base_ds))
    else:
        img_idx = int(max(0, min(len(ds_full.base_ds) - 1, image_index)))

    # Locate sample index for (img_idx, k_value)
    sample_idx = None
    for i, (im_i, k) in enumerate(ds_full.index_map):
        if im_i == img_idx and int(k) == int(k_value):
            sample_idx = i
            break
    if sample_idx is None:
        print(f"[plot] Could not find sample for chosen image and k={k_value}")
        return

    sample = ds_full[sample_idx]
    x = sample["x"].to(device)
    if normalize_node_features:
        x = F.normalize(x, p=2, dim=1)
    edge_index = sample["edge_index"].to(device)
    y_counts = sample["y"].cpu().numpy()  # [N,C]

    # Pick the superpixel (node) with the most labels present (count > 0); break ties by total pixels
    labels_present = (y_counts > 0).sum(axis=1)
    totals = y_counts.sum(axis=1)
    best_idx = int(np.lexsort((-totals, labels_present))[-1])  # sort by labels_present asc, totals asc; take last

    # Forward models and get probabilities for this node
    eff_palette = _effective_palette(class_rgb_values, unknown_index)
    probs: List[np.ndarray] = []
    with torch.inference_mode():
        for model in model_list:
            logits = model(x, edge_index)
            p = F.softmax(logits, dim=-1)[best_idx].detach().cpu().numpy()
            probs.append(p)

    # Stacked bars of unit height for each model
    colors = [tuple(np.array(c) / 255.0) for c in eff_palette]
    class_names_eff = [
        name for i, name in enumerate(class_rgb_values)
        if not (unknown_index is not None and i == unknown_index)
    ]
    # class_names_eff are RGB tuples; for legend we will build patches with index labels
    x_pos = np.arange(len(probs))
    width = 0.6
    plt.figure(figsize=(max(6, 3 * len(probs)), 4))
    for i, p in enumerate(probs):
        bottom = 0.0
        for c_idx in range(len(p)):
            h = float(p[c_idx])
            if h <= 0:
                continue
            plt.bar(x_pos[i], h, width=width, bottom=bottom, color=colors[c_idx], edgecolor='black', linewidth=0.2)
            bottom += h
        # Ensure the bar reaches 1 unit for visual consistency (in case of numeric issues)
        if bottom < 1.0:
            plt.bar(x_pos[i], 1.0 - bottom, width=width, bottom=bottom, color=(0.95, 0.95, 0.95), edgecolor='black', linewidth=0.2)

    plt.xticks(x_pos, model_labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Predicted fraction (unit height)")
    img_stem = Path(sample["meta"]["image_path"]).stem
    plt.title(f"Superpixel label distribution: {img_stem} (k={k_value}, node={best_idx})")

    # Build legend from top-K average important classes to avoid clutter, or all classes if small
    try:
        import matplotlib.patches as mpatches
        patches = []
        # Rank classes by average contribution across models for this node
        mean_contrib = np.mean(np.stack(probs, axis=0), axis=0)
        order = np.argsort(-mean_contrib)
        max_legend = min(len(order), 10)
        for idx in order[:max_legend]:
            name = f"c{idx}"
            patches.append(mpatches.Patch(color=colors[idx], label=name))
        if patches:
            plt.legend(handles=patches, title="Top classes", bbox_to_anchor=(1.02, 1), loc='upper left')
    except Exception:
        pass

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"superpixel_distribution_{img_stem}_k{k_value}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def _plot_metrics_comparison(
    save_dir: str,
    model_labels: List[str],
    classification_metrics: List[Tuple[float, float, float, float]],  # (p,r,f1,f2)
    node_macro_metrics: List[dict],  # keys: mae, rmse, brier, js, hell, cos, r2, pear, spear
    image_macro_metrics: List[dict],  # keys: mae, rmse, r2, smape, dice, js, emd
    per_class_names: List[str],
    per_class_r2_nodes: List[List[float]],
    per_class_dice_images: List[List[float]],
    show: bool,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # 1) Classification scalars
    plt.figure(figsize=(8, 4))
    labels = ["Precision", "Recall", "F1", "F2"]
    # Series per model: values across the 4 metrics
    series = [[cm[0], cm[1], cm[2], cm[3]] for cm in classification_metrics]
    ax = plt.gca()
    _grouped_bar(ax, labels, series, model_labels)
    ax.set_title("Classification metrics (test)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "classification_metrics.png"), dpi=150)
    if show:
        plt.show()
    plt.close()

    # 2) Node-level macro metrics
    node_labels = ["MAE", "RMSE", "Brier", "JS", "Hellinger", "Cosine", "R2", "Pearson", "Spearman"]
    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    # Series per model
    key_map = {
        "MAE": "mae", "RMSE": "rmse", "Brier": "brier", "JS": "js", "Hellinger": "hell",
        "Cosine": "cos", "R2": "r2", "Pearson": "pear", "Spearman": "spear"
    }
    series = []
    for nm in node_macro_metrics:
        series.append([nm[key_map[lab]] for lab in node_labels])
    _grouped_bar(ax, node_labels, series, model_labels)
    ax.set_title("Node-level macro metrics (test)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "node_macro_metrics.png"), dpi=150)
    if show:
        plt.show()
    plt.close()

    # 3) Image-level macro metrics
    img_labels = ["MAE", "RMSE", "R2", "sMAPE", "Dice", "JS", "EMD"]
    keys = ["mae", "rmse", "r2", "smape", "dice", "js", "emd"]
    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    # Series per model
    series = []
    for im in image_macro_metrics:
        series.append([im[k] for k in keys])
    _grouped_bar(ax, img_labels, series, model_labels)
    ax.set_title("Image-level macro metrics (test)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "image_macro_metrics.png"), dpi=150)
    if show:
        plt.show()
    plt.close()

    # 4) Per-class R2 (nodes)
    if len(per_class_names) and len(per_class_r2_nodes):
        plt.figure(figsize=(max(8, 0.4 * len(per_class_names) * max(1, len(model_labels))), 4))
        ax = plt.gca()
        _grouped_bar(ax, per_class_names, per_class_r2_nodes, model_labels)
        ax.set_ylabel("R2")
        ax.set_title("Per-class R2 (nodes)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "per_class_r2_nodes.png"), dpi=150)
        if show:
            plt.show()
        plt.close()

    # 5) Per-class Dice (images)
    if len(per_class_names) and len(per_class_dice_images):
        plt.figure(figsize=(max(8, 0.4 * len(per_class_names) * max(1, len(model_labels))), 4))
        ax = plt.gca()
        _grouped_bar(ax, per_class_names, per_class_dice_images, model_labels)
        ax.set_ylabel("Soft Dice")
        ax.set_title("Per-class Dice (images)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "per_class_dice_images.png"), dpi=150)
        if show:
            plt.show()
        plt.close()


def main(cfg: EvalConfig) -> None:
    # Backend tuning for eval throughput
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    # Initialize logger (no-op if disabled)
    logger = TrainLogger(use_wandb=cfg.use_wandb, project=cfg.wandb_project, run_name=(f"eval-{Path(cfg.ckpt_path).stem}" if cfg.ckpt_path else "eval"))

    # Shared backbone for all splits, built once
    backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(cfg.feature_device).eval()
    img_size = None
    if cfg.img_size_w is not None and cfg.img_size_h is not None:
        img_size = (int(cfg.img_size_w), int(cfg.img_size_h))

    # Build dataset from train root; derive deterministic val split from checkpoint JSON if present
    ds_full, names, class_rgb_values, unknown_index = make_dataset(cfg.train_dir, cfg.class_csv, img_size, cfg.k_values, cfg.feature_device, cfg.cache_dir, normalize_targets=False, backbone=backbone, device=cfg.device, hsv_threshold=cfg.hsv_threshold)

    num_classes_eff = len(class_rgb_values) - (1 if (unknown_index is not None and 0 <= unknown_index < len(class_rgb_values)) else 0)

    # Load model(s) and read sidecar JSON for val split info if present
    model = load_model(cfg, num_classes_eff)
    model2: Optional[SPNodeRegressor] = None
    if cfg.ckpt2_path:
        cfg2 = EvalConfig(
            train_dir=cfg.train_dir,
            valid_dir=cfg.valid_dir,
            test_dir=cfg.test_dir,
            class_csv=cfg.class_csv,
            img_size_w=cfg.img_size_w,
            img_size_h=cfg.img_size_h,
            k_values=cfg.k_values,
            cache_dir=cfg.cache_dir,
            feature_device=cfg.feature_device,
            hsv_threshold=cfg.hsv_threshold,
            feature_batch_size=cfg.feature_batch_size,
            slic_backend=cfg.slic_backend,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            in_dim=cfg.in_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            gat_dropout=cfg.gat_dropout,
            integrate_dropout=cfg.integrate_dropout,
            normalize_node_features=cfg.normalize_node_features,
            device=cfg.device,
            seed=cfg.seed,
            ckpt_path=cfg.ckpt2_path,
            ckpt2_path=None,
            plot_sample=cfg.plot_sample,
            plot_k=cfg.plot_k,
            save_plots_dir=cfg.save_plots_dir,
            no_show=cfg.no_show,
        )
        model2 = load_model(cfg2, num_classes_eff)
    arch_meta_path = Path(cfg.ckpt_path).with_suffix('.json') if cfg.ckpt_path else None
    val_image_ids: Optional[List[str]] = None
    if arch_meta_path is not None and arch_meta_path.exists():
        try:
            with open(arch_meta_path, 'r') as f:
                meta = json.load(f)
            val_image_ids = meta.get("val_image_ids")
        except Exception:
            val_image_ids = None

    # Derive subsets by image indices
    image_paths = ds_full.base_ds.image_paths
    stem_to_imgidx = {Path(p).stem: i for i, p in enumerate(image_paths)}
    if val_image_ids:
        val_img_indices = sorted([stem_to_imgidx[s] for s in val_image_ids if s in stem_to_imgidx])
    else:
        # Fallback: use a default split seed and fraction similar to training defaults
        import random
        rng = random.Random(cfg.seed)
        img_indices = list(range(len(image_paths)))
        rng.shuffle(img_indices)
        n_val = max(1, int(round(0.1 * len(img_indices))))
        val_img_indices = sorted(img_indices[:n_val])

    val_img_set = set(val_img_indices)
    val_sample_indices = [i for i, (img_idx, _k) in enumerate(ds_full.index_map) if img_idx in val_img_set]
    test_sample_indices = [i for i, (img_idx, _k) in enumerate(ds_full.index_map) if img_idx not in val_img_set]

    ds_valid = torch.utils.data.Subset(ds_full, val_sample_indices)
    ds_test = torch.utils.data.Subset(ds_full, test_sample_indices)

    # DataLoaders
    # Avoid CUDA-in-fork: if feature extraction runs on CUDA, don't use worker processes
    try:
        if torch.device(cfg.feature_device).type == "cuda" and (cfg.num_workers is None or cfg.num_workers > 0):
            print("[eval] Detected CUDA feature extraction with DataLoader workers > 0; setting num_workers=0 to avoid CUDA-in-fork.")
            cfg.num_workers = 0
    except Exception:
        pass
    pin_memory = (torch.device(cfg.device).type == "cuda")
    valid_loader = DataLoader(ds_valid, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_graphs, pin_memory=pin_memory, persistent_workers=(cfg.num_workers > 0), prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None))
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_graphs, pin_memory=pin_memory, persistent_workers=(cfg.num_workers > 0), prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None))

    # Collect validation scores and calibrate thresholds on valid only (classification-style)
    y_score_val, y_true_val = collect_image_scores(model, valid_loader, cfg.normalize_node_features)
    thresholds = calibrate_thresholds(y_score_val, y_true_val, beta=2.0)
    thresholds2 = None
    if model2 is not None:
        y_score_val2, y_true_val2 = collect_image_scores(model2, valid_loader, cfg.normalize_node_features)
        thresholds2 = calibrate_thresholds(y_score_val2, y_true_val2, beta=2.0)

    # Evaluate on test (classification-style summary)
    y_score_test, y_true_test = collect_image_scores(model, test_loader, cfg.normalize_node_features)
    p, r, f1, f2 = example_based_metrics(y_true_test, y_score_test, thresholds)
    print("Classification-style (image presence) metrics on test:")
    print(f"Precision: {p*100:.2f}%  Recall: {r*100:.2f}%  F1: {f1*100:.2f}%  F2: {f2*100:.2f}%")
    p2 = r2 = f12 = f22 = None
    if model2 is not None and thresholds2 is not None:
        y_score_test2, y_true_test2 = collect_image_scores(model2, test_loader, cfg.normalize_node_features)
        p2, r2, f12, f22 = example_based_metrics(y_true_test2, y_score_test2, thresholds2)
        print("Classification-style metrics on test (Model 2):")
        print(f"Precision: {p2*100:.2f}%  Recall: {r2*100:.2f}%  F1: {f12*100:.2f}%  F2: {f22*100:.2f}%")

    # Collect regression data (node-level and image-level distributions)
    reg = collect_regression_data(model, test_loader, cfg.normalize_node_features)
    nodes = reg["nodes"]
    images = reg["images"]

    # Containers for plotting
    class_labels_eff = [n for i, n in enumerate(names) if not (unknown_index is not None and i == unknown_index)]
    node_macro_plot = None
    img_macro_plot = None
    per_class_r2_plot = None
    per_class_dice_plot = None

    if nodes["y_true"].numel() > 0:
        y_true_nodes = nodes["y_true"].numpy()
        y_pred_nodes = nodes["y_pred"].numpy()
        w_nodes = nodes["weights"].numpy()

        node_mae = mae_weighted_nodes(y_true_nodes, y_pred_nodes, w_nodes)
        node_rmse = rmse_weighted_nodes(y_true_nodes, y_pred_nodes, w_nodes)
        node_brier = brier_score_weighted_nodes(y_true_nodes, y_pred_nodes, w_nodes)
        node_js = _js_divergence_batch(y_true_nodes, y_pred_nodes, w_nodes)
        node_hell = _hellinger_batch(y_true_nodes, y_pred_nodes, w_nodes)
        node_cos = _cosine_sim_batch(y_true_nodes, y_pred_nodes, w_nodes)
        r2_pc, r2_macro = r2_per_class_nodes(y_true_nodes, y_pred_nodes, w_nodes)
        pear_pc, spear_pc, pear_macro, spear_macro = pearson_spearman_per_class_nodes(y_true_nodes, y_pred_nodes)

        print("\nNode-level regression metrics (area-weighted):")
        print(f"MAE: {node_mae:.6f}  RMSE: {node_rmse:.6f}  Brier: {node_brier:.6f}")
        print(f"Jensen-Shannon: {node_js:.6f}  Hellinger: {node_hell:.6f}  Cosine: {node_cos:.6f}")
        print(f"R2 per class (macro): {r2_macro:.4f}")
        # Optionally print per-class R2/correlations concisely
        # print(f"R2 per class: {r2_pc}")
        print(f"Pearson per class (macro): {pear_macro:.4f}  Spearman per class (macro): {spear_macro:.4f}")

        node_macro_plot = {
            "mae": float(node_mae),
            "rmse": float(node_rmse),
            "brier": float(node_brier),
            "js": float(node_js),
            "hell": float(node_hell),
            "cos": float(node_cos),
            "r2": float(r2_macro),
            "pear": float(pear_macro),
            "spear": float(spear_macro),
        }
        per_class_r2_plot = [float(r2_pc[c]) for c in range(len(class_labels_eff))]

    if images["y_true"].numel() > 0:
        y_true_img = images["y_true"].numpy()
        y_pred_img = images["y_pred"].numpy()
        w_img = images["weights"].numpy()

        img_scores = per_class_regression_scores_images(y_true_img, y_pred_img, w_img)
        dice_pc, dice_macro = soft_dice_per_class(y_true_img, y_pred_img)
        js_img = js_divergence_images(y_true_img, y_pred_img)
        emd_img = emd_1d_images(y_true_img, y_pred_img)

        print("\nImage-level regression metrics:")
        print(f"Macro MAE: {img_scores['macro']['mae']:.6f}  Macro RMSE: {img_scores['macro']['rmse']:.6f}  Macro R2: {img_scores['macro']['r2']:.4f}  Macro sMAPE: {img_scores['macro']['smape']:.2f}%")
        print(f"Micro MAE: {img_scores['micro']['mae']:.6f}  Micro RMSE: {img_scores['micro']['rmse']:.6f}")
        print(f"Soft Dice per class (macro): {dice_macro:.6f}  JS-divergence (avg): {js_img:.6f}  1D-EMD (avg): {emd_img:.6f}")

        img_macro_plot = {
            "mae": float(img_scores["macro"]["mae"]),
            "rmse": float(img_scores["macro"]["rmse"]),
            "r2": float(img_scores["macro"]["r2"]),
            "smape": float(img_scores["macro"]["smape"]),
            "dice": float(dice_macro),
            "js": float(js_img),
            "emd": float(emd_img),
        }
        per_class_dice_plot = [float(dice_pc[c]) for c in range(len(class_labels_eff))]

    # Repeat metrics collection for model2 if provided
    class_metrics = [(p, r, f1, f2)]
    node_macro_metrics = [node_macro_plot] if node_macro_plot is not None else []
    image_macro_metrics = [img_macro_plot] if img_macro_plot is not None else []
    per_class_r2_all = [per_class_r2_plot] if per_class_r2_plot is not None else []
    per_class_dice_all = [per_class_dice_plot] if per_class_dice_plot is not None else []

    if model2 is not None:
        reg2 = collect_regression_data(model2, test_loader, cfg.normalize_node_features)
        nodes2 = reg2["nodes"]
        images2 = reg2["images"]

        # Node-level
        if nodes2["y_true"].numel() > 0:
            y_true_nodes2 = nodes2["y_true"].numpy()
            y_pred_nodes2 = nodes2["y_pred"].numpy()
            w_nodes2 = nodes2["weights"].numpy()
            node_mae2 = mae_weighted_nodes(y_true_nodes2, y_pred_nodes2, w_nodes2)
            node_rmse2 = rmse_weighted_nodes(y_true_nodes2, y_pred_nodes2, w_nodes2)
            node_brier2 = brier_score_weighted_nodes(y_true_nodes2, y_pred_nodes2, w_nodes2)
            node_js2 = _js_divergence_batch(y_true_nodes2, y_pred_nodes2, w_nodes2)
            node_hell2 = _hellinger_batch(y_true_nodes2, y_pred_nodes2, w_nodes2)
            node_cos2 = _cosine_sim_batch(y_true_nodes2, y_pred_nodes2, w_nodes2)
            r2_pc2, r2_macro2 = r2_per_class_nodes(y_true_nodes2, y_pred_nodes2, w_nodes2)
            pear_pc2, spear_pc2, pear_macro2, spear_macro2 = pearson_spearman_per_class_nodes(y_true_nodes2, y_pred_nodes2)
            node_macro_metrics.append({
                "mae": float(node_mae2),
                "rmse": float(node_rmse2),
                "brier": float(node_brier2),
                "js": float(node_js2),
                "hell": float(node_hell2),
                "cos": float(node_cos2),
                "r2": float(r2_macro2),
                "pear": float(pear_macro2),
                "spear": float(spear_macro2),
            })
            per_class_r2_all.append([float(r2_pc2[c]) for c in range(len(class_labels_eff))])

        # Image-level
        if images2["y_true"].numel() > 0:
            y_true_img2 = images2["y_true"].numpy()
            y_pred_img2 = images2["y_pred"].numpy()
            w_img2 = images2["weights"].numpy()
            img_scores2 = per_class_regression_scores_images(y_true_img2, y_pred_img2, w_img2)
            dice_pc2, dice_macro2 = soft_dice_per_class(y_true_img2, y_pred_img2)
            js_img2 = js_divergence_images(y_true_img2, y_pred_img2)
            emd_img2 = emd_1d_images(y_true_img2, y_pred_img2)
            image_macro_metrics.append({
                "mae": float(img_scores2["macro"]["mae"]),
                "rmse": float(img_scores2["macro"]["rmse"]),
                "r2": float(img_scores2["macro"]["r2"]),
                "smape": float(img_scores2["macro"]["smape"]),
                "dice": float(dice_macro2),
                "js": float(js_img2),
                "emd": float(emd_img2),
            })
            per_class_dice_all.append([float(dice_pc2[c]) for c in range(len(class_labels_eff))])

        if p2 is not None:
            class_metrics.append((p2, r2, f12, f22))

    # Optional plotting
    if cfg.plot_sample:
        # Honor specific k if provided, else use first available
        k_for_plot = int(cfg.plot_k) if (cfg.plot_k is not None) else int(ds_full.k_values[0])
        if k_for_plot not in ds_full.k_values:
            print(f"[plot] Requested k={k_for_plot} not in dataset k_values={ds_full.k_values}; using {ds_full.k_values[0]}")
            k_for_plot = int(ds_full.k_values[0])
        labels = [Path(cfg.ckpt_path).stem]
        models = [model]
        if model2 is not None:
            labels.append(Path(cfg.ckpt2_path).stem)
            models.append(model2)
        # Resolve image index if specified
        img_index_for_plot: Optional[int] = None
        if cfg.plot_image is not None:
            # Accept numeric index or image stem
            try:
                img_index_for_plot = int(cfg.plot_image)
            except Exception:
                stems = [Path(p).stem for p in ds_full.base_ds.image_paths]
                if cfg.plot_image in stems:
                    img_index_for_plot = stems.index(cfg.plot_image)
                else:
                    print(f"[plot] plot_image='{cfg.plot_image}' not found; using random image")
        _plot_random_sample_for_k(
            ds_full=ds_full,
            model_list=models,
            model_labels=labels,
            names=names,
            class_rgb_values=class_rgb_values,
            unknown_index=unknown_index,
            k_value=k_for_plot,
            normalize_node_features=cfg.normalize_node_features,
            device=next(model.parameters()).device,
            save_dir=cfg.save_plots_dir,
            show=(not cfg.no_show),
            image_index=img_index_for_plot,
        )
        _plot_superpixel_distribution_for_k(
            ds_full=ds_full,
            model_list=models,
            model_labels=labels,
            class_rgb_values=class_rgb_values,
            unknown_index=unknown_index,
            k_value=k_for_plot,
            normalize_node_features=cfg.normalize_node_features,
            device=next(model.parameters()).device,
            save_dir=cfg.save_plots_dir,
            show=(not cfg.no_show),
            image_index=img_index_for_plot,
        )

    # Metrics comparison plots (if we have at least one set)
    if len(class_metrics) >= 1 and (len(node_macro_metrics) >= 1 or len(image_macro_metrics) >= 1):
        model_labels = [Path(cfg.ckpt_path).stem]
        if model2 is not None:
            model_labels.append(Path(cfg.ckpt2_path).stem)
        # Normalize lengths in case some parts missing
        while len(node_macro_metrics) < len(model_labels):
            node_macro_metrics.append(node_macro_metrics[-1])
        while len(image_macro_metrics) < len(model_labels):
            image_macro_metrics.append(image_macro_metrics[-1])
        while len(class_metrics) < len(model_labels):
            class_metrics.append(class_metrics[-1])
        _plot_metrics_comparison(
            save_dir=cfg.save_plots_dir,
            model_labels=model_labels,
            classification_metrics=class_metrics,
            node_macro_metrics=node_macro_metrics,
            image_macro_metrics=image_macro_metrics,
            per_class_names=class_labels_eff,
            per_class_r2_nodes=per_class_r2_all,
            per_class_dice_images=per_class_dice_all,
            show=(not cfg.no_show),
        )

        # Log plots to Weights & Biases both as images and as data tables
        if cfg.use_wandb:
            try:
                import wandb  # type: ignore
                # 1) Classification metrics table
                cls_table = wandb.Table(columns=["model", "precision", "recall", "f1", "f2"])
                for m_label, (pp, rr, ff1, ff2) in zip(model_labels, class_metrics):
                    cls_table.add_data(m_label, float(pp), float(rr), float(ff1), float(ff2))
                logger.log({"eval/tables/classification_metrics": cls_table})

                # 2) Node-level macro metrics table
                if len(node_macro_metrics) >= 1:
                    node_table = wandb.Table(columns=["model", "mae", "rmse", "brier", "js", "hell", "cos", "r2", "pearson", "spearman"])
                    for m_label, nm in zip(model_labels, node_macro_metrics):
                        node_table.add_data(
                            m_label,
                            float(nm.get("mae", float("nan"))),
                            float(nm.get("rmse", float("nan"))),
                            float(nm.get("brier", float("nan"))),
                            float(nm.get("js", float("nan"))),
                            float(nm.get("hell", float("nan"))),
                            float(nm.get("cos", float("nan"))),
                            float(nm.get("r2", float("nan"))),
                            float(nm.get("pear", float("nan"))),
                            float(nm.get("spear", float("nan"))),
                        )
                    logger.log({"eval/tables/node_macro_metrics": node_table})

                # 3) Image-level macro metrics table
                if len(image_macro_metrics) >= 1:
                    img_table = wandb.Table(columns=["model", "mae", "rmse", "r2", "smape", "dice", "js", "emd"])
                    for m_label, im in zip(model_labels, image_macro_metrics):
                        img_table.add_data(
                            m_label,
                            float(im.get("mae", float("nan"))),
                            float(im.get("rmse", float("nan"))),
                            float(im.get("r2", float("nan"))),
                            float(im.get("smape", float("nan"))),
                            float(im.get("dice", float("nan"))),
                            float(im.get("js", float("nan"))),
                            float(im.get("emd", float("nan"))),
                        )
                    logger.log({"eval/tables/image_macro_metrics": img_table})

                # 4) Per-class tables (R2 for nodes, Dice for images)
                if len(per_class_r2_all) >= 1 and len(class_labels_eff) >= 1:
                    pc_r2_cols = ["class"] + list(model_labels)
                    pc_r2_table = wandb.Table(columns=pc_r2_cols)
                    for ci, cname in enumerate(class_labels_eff):
                        row = [str(cname)] + [float(r2_list[ci]) for r2_list in per_class_r2_all]
                        pc_r2_table.add_data(*row)
                    logger.log({"eval/tables/per_class_r2_nodes": pc_r2_table})
                if len(per_class_dice_all) >= 1 and len(class_labels_eff) >= 1:
                    pc_dice_cols = ["class"] + list(model_labels)
                    pc_dice_table = wandb.Table(columns=pc_dice_cols)
                    for ci, cname in enumerate(class_labels_eff):
                        row = [str(cname)] + [float(dice_list[ci]) for dice_list in per_class_dice_all]
                        pc_dice_table.add_data(*row)
                    logger.log({"eval/tables/per_class_dice_images": pc_dice_table})

                # 5) Log the generated plot images
                plot_files = [
                    os.path.join(cfg.save_plots_dir, "classification_metrics.png"),
                    os.path.join(cfg.save_plots_dir, "node_macro_metrics.png"),
                    os.path.join(cfg.save_plots_dir, "image_macro_metrics.png"),
                ]
                # Conditionally present plots
                r2_plot = os.path.join(cfg.save_plots_dir, "per_class_r2_nodes.png")
                dice_plot = os.path.join(cfg.save_plots_dir, "per_class_dice_images.png")
                if os.path.exists(r2_plot):
                    plot_files.append(r2_plot)
                if os.path.exists(dice_plot):
                    plot_files.append(dice_plot)

                img_payload = {}
                for pth in plot_files:
                    if os.path.exists(pth):
                        key = f"eval/plots/{Path(pth).name.replace('.png','')}"
                        img_payload[key] = wandb.Image(pth)
                if len(img_payload) > 0:
                    logger.log(img_payload)

                # 6) Log sample visualization plots if present
                # sample_k*.png and superpixel_distribution_*.png
                extra_imgs = []
                for fname in os.listdir(cfg.save_plots_dir):
                    if fname.startswith("sample_k") and fname.endswith(".png"):
                        extra_imgs.append(os.path.join(cfg.save_plots_dir, fname))
                    if fname.startswith("superpixel_distribution_") and fname.endswith(".png"):
                        extra_imgs.append(os.path.join(cfg.save_plots_dir, fname))
                extra_payload = {}
                for pth in extra_imgs:
                    extra_payload[f"eval/plots/{Path(pth).name.replace('.png','')}"] = wandb.Image(pth)
                if len(extra_payload) > 0:
                    logger.log(extra_payload)
            except Exception:
                pass

    # Finish logger
    try:
        logger.finish()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate GAT-based superpixel regressor with paper metrics")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    p.add_argument("--ckpt2", type=str, default=None, help="Optional second checkpoint to compare")
    p.add_argument("--train_dir", type=str, default=None)
    p.add_argument("--valid_dir", type=str, default=None)
    p.add_argument("--test_dir", type=str, default=None)
    p.add_argument("--class_csv", type=str, default=None)
    p.add_argument("--img_size_w", type=int, default=None)
    p.add_argument("--img_size_h", type=int, default=None)
    p.add_argument("--k_values", type=int, nargs="+", default=None, help="One or more k values for SLIC. Use with --plot_k to select a specific k to visualize.")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--feature_device", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--prefetch_factor", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--hsv_threshold", type=float, default=None)
    p.add_argument("--slic_backend", type=str, default=None, choices=["cpu"])  # fixed to skimage
    # Plot flags
    p.add_argument("--plot_sample", action="store_true", help="Plot a random sample (image/mask/predictions)")
    p.add_argument("--plot_k", type=int, default=None, help="Specific k to visualize (must be in --k_values)")
    p.add_argument("--save_plots_dir", type=str, default=None, help="Directory to save plots")
    p.add_argument("--no_show", action="store_true", help="Do not display plots interactively; save only")
    p.add_argument("--plot_image", type=str, default=None, help="Image index or stem to use for plotting")
    p.add_argument("--use_wandb", action="store_true", help="Log results to Weights & Biases, including plots as images and data tables")
    p.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = EvalConfig()
    cfg.ckpt_path = args.ckpt
    cfg.ckpt2_path = args.ckpt2
    if args.train_dir is not None: cfg.train_dir = args.train_dir
    if args.valid_dir is not None: cfg.valid_dir = args.valid_dir
    if args.test_dir is not None: cfg.test_dir = args.test_dir
    if args.class_csv is not None: cfg.class_csv = args.class_csv
    if args.img_size_w is not None: cfg.img_size_w = args.img_size_w
    if args.img_size_h is not None: cfg.img_size_h = args.img_size_h
    if args.k_values is not None: cfg.k_values = args.k_values
    if args.cache_dir is not None: cfg.cache_dir = args.cache_dir
    if args.feature_device is not None: cfg.feature_device = args.feature_device
    if args.device is not None: cfg.device = args.device
    if args.num_workers is not None: cfg.num_workers = args.num_workers
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.prefetch_factor is not None: cfg.prefetch_factor = args.prefetch_factor
    if args.seed is not None: cfg.seed = args.seed
    if args.hsv_threshold is not None: cfg.hsv_threshold = args.hsv_threshold
    if args.slic_backend is not None: cfg.slic_backend = args.slic_backend
    if args.plot_sample: cfg.plot_sample = True
    if args.plot_k is not None: cfg.plot_k = args.plot_k
    if args.save_plots_dir is not None: cfg.save_plots_dir = args.save_plots_dir
    if args.no_show: cfg.no_show = True
    if args.plot_image is not None: cfg.plot_image = args.plot_image
    if args.use_wandb: cfg.use_wandb = True
    if args.wandb_project is not None: cfg.wandb_project = args.wandb_project
    main(cfg)


