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

from load_palette import load_class_palette
from datasets.graph_superpixel_dataset import GraphSuperpixelDataset
from models.gat import SPNodeRegressor
from train import collate_graphs as train_collate_graphs
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
    batch_size: int = 128
    num_workers: int = 4
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

    # Checkpoint
    ckpt_path: Optional[str] = None


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
        precompute=True,
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
    # Shared backbone for all splits, built once
    backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(cfg.feature_device).eval()
    img_size = None
    if cfg.img_size_w is not None and cfg.img_size_h is not None:
        img_size = (int(cfg.img_size_w), int(cfg.img_size_h))

    # Build dataset from train root; derive deterministic val split from checkpoint JSON if present
    ds_full, names, class_rgb_values, unknown_index = make_dataset(cfg.train_dir, cfg.class_csv, img_size, cfg.k_values, cfg.feature_device, cfg.cache_dir, normalize_targets=False, backbone=backbone, device=cfg.device, hsv_threshold=cfg.hsv_threshold)

    num_classes_eff = len(class_rgb_values) - (1 if (unknown_index is not None and 0 <= unknown_index < len(class_rgb_values)) else 0)

    # Load model and read sidecar JSON for val split info if present
    model = load_model(cfg, num_classes_eff)
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
    pin_memory = (torch.device(cfg.device).type == "cuda")
    valid_loader = DataLoader(ds_valid, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_graphs, pin_memory=pin_memory, persistent_workers=(cfg.num_workers > 0), prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None))
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_graphs, pin_memory=pin_memory, persistent_workers=(cfg.num_workers > 0), prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None))

    # Collect validation scores and calibrate thresholds on valid only (classification-style)
    y_score_val, y_true_val = collect_image_scores(model, valid_loader, cfg.normalize_node_features)
    thresholds = calibrate_thresholds(y_score_val, y_true_val, beta=2.0)

    # Evaluate on test (classification-style summary)
    y_score_test, y_true_test = collect_image_scores(model, test_loader, cfg.normalize_node_features)
    p, r, f1, f2 = example_based_metrics(y_true_test, y_score_test, thresholds)
    print("Classification-style (image presence) metrics on test:")
    print(f"Precision: {p*100:.2f}%  Recall: {r*100:.2f}%  F1: {f1*100:.2f}%  F2: {f2*100:.2f}%")

    # Collect regression data (node-level and image-level distributions)
    reg = collect_regression_data(model, test_loader, cfg.normalize_node_features)
    nodes = reg["nodes"]
    images = reg["images"]

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate GAT-based superpixel regressor with paper metrics")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    p.add_argument("--train_dir", type=str, default=None)
    p.add_argument("--valid_dir", type=str, default=None)
    p.add_argument("--test_dir", type=str, default=None)
    p.add_argument("--class_csv", type=str, default=None)
    p.add_argument("--img_size_w", type=int, default=None)
    p.add_argument("--img_size_h", type=int, default=None)
    p.add_argument("--k_values", type=int, nargs="+", default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--feature_device", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--prefetch_factor", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--hsv_threshold", type=float, default=None)
    p.add_argument("--slic_backend", type=str, default=None, choices=["cpu"])  # fixed to skimage
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = EvalConfig()
    cfg.ckpt_path = args.ckpt
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
    main(cfg)


