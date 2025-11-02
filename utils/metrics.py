from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# -----------------------------
# Classification-oriented helpers
# -----------------------------
def image_scores_from_nodes(logits: torch.Tensor, y_counts: torch.Tensor) -> torch.Tensor:
    prob = F.softmax(logits, dim=-1)
    weights = y_counts.sum(dim=-1)
    weights = torch.where(weights > 0, weights, torch.ones_like(weights))
    s = (prob * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()
    return s


def image_level_ground_truth(mask_counts: torch.Tensor) -> torch.Tensor:
    per_class_total = mask_counts.sum(dim=0)
    return (per_class_total > 0).float()


def collect_image_scores(model: torch.nn.Module, loader: DataLoader, normalize_node_features: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    y_scores: List[torch.Tensor] = []
    y_true: List[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            if normalize_node_features:
                x = F.normalize(x, p=2, dim=1)
            edge_index = batch["edge_index"].to(device, non_blocking=True)
            y_counts = batch["y"].to(device, non_blocking=True)
            logits = model(x, edge_index)
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
def collect_regression_data(model: torch.nn.Module, loader: DataLoader, normalize_node_features: bool) -> dict:
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
            y_counts = batch["y"].to(device, non_blocking=True)
            logits = model(x, edge_index)
            prob = F.softmax(logits, dim=-1)

            meta = batch.get("meta", {}) if isinstance(batch, dict) else {}
            npg = meta.get("nodes_per_graph", None)
            if not isinstance(npg, list) or len(npg) == 0:
                npg = [x.shape[0]]
            offset = 0
            for n in npg:
                n = int(n)
                prob_g = prob[offset:offset+n]
                counts_g = y_counts[offset:offset+n]

                true_frac_g, valid_mask_g = _safe_row_normalize(counts_g)
                weights_g = counts_g.sum(dim=-1)

                if valid_mask_g.any():
                    pred_nodes.append(prob_g[valid_mask_g].detach().cpu())
                    true_nodes.append(true_frac_g[valid_mask_g].detach().cpu())
                    node_weights.append(weights_g[valid_mask_g].detach().cpu())

                total_pixels = weights_g.sum()
                if float(total_pixels.item()) > 0:
                    pred_counts_img = (prob_g * weights_g.unsqueeze(-1)).sum(dim=0)
                    true_counts_img = counts_g.sum(dim=0)
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
    js = 0.5 * (kl_pm + kl_qm)
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
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
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
    vals = []
    for i in range(y_true.shape[0]):
        vals.append(_js_divergence_batch(y_true[i:i+1], y_pred[i:i+1]))
    return float(np.mean(vals)) if len(vals) else float("nan")


def emd_1d_images(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    def emd_1d(p: np.ndarray, q: np.ndarray) -> float:
        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)
        return float(np.sum(np.abs(cdf_p - cdf_q)))
    vals = [emd_1d(y_true[i], y_pred[i]) for i in range(y_true.shape[0])]
    return float(np.mean(vals)) if len(vals) else float("nan")


def per_class_regression_scores_images(y_true: np.ndarray, y_pred: np.ndarray, image_weights: Optional[np.ndarray] = None) -> dict:
    C = y_true.shape[1]
    out = {"per_class": {}, "macro": {}, "micro": {}}
    mae_vals = []
    rmse_vals = []
    r2_vals = []
    smape_vals = []
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


