import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]

import const
from dataset_loader import DeepGlobeDataset
from datasets.superpixel_graph_dataset_v2 import SuperpixelGraphDatasetV2
from load_palette import load_class_palette
from models.gat_geometric import MultiheadGAT
from models.gcn import GCN2
from models.transformer import Transformer
from utils.metrics import (
    collect_image_scores,
    collect_regression_data,
    example_based_metrics,
    calibrate_thresholds,
    mae_weighted_nodes,
    rmse_weighted_nodes,
    brier_score_weighted_nodes,
    js_divergence_images,
    emd_1d_images,
    per_class_regression_scores_images,
    pearson_spearman_per_class_nodes,
    r2_per_class_nodes,
    soft_dice_per_class,
    _js_divergence_batch,
    _hellinger_batch,
)

plt.switch_backend("Agg")

MODEL_CLASS_MAP = {
    "Transformer": Transformer,
    "GCN2": GCN2,
    "MultiheadGAT": MultiheadGAT,
}


# Default Hugging Face repositories for pre-trained checkpoints.
# Update the repo ids with the actual namespaces once the models are uploaded.
HF_MODEL_REGISTRY: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {
    "GCN2": {
        "single_k": {
            "repo_id": "mojique/gcn-single-k",
            "checkpoint": "gcn2_k60_best.pt",
            "config": "gcn2_k60_best.json",
            "revision": None,
        },
        "multi_k": {
            "repo_id": "mojique/gcn-multi-k",
            "checkpoint": "gcn2_k60_k120_k300_k500_multi_best.pt",
            "config": "gcn2_k60_k120_k300_k500_multi_best.json",
            "revision": None,
        },
    },
    "Transformer": {
        "single_k": {
            "repo_id": "mojique/transformer-single-k",
            "checkpoint": "transformer_k60_best.pt",
            "config": "transformer_k60_best.json",
            "revision": None,
        },
        "multi_k": {
            "repo_id": "mojique/transformer-multi-k",
            "checkpoint": "transformer_k60_k120_k300_k500_best.pt",
            "config": "transformer_k60_k120_k300_k500_best.json",
            "revision": None,
        },
    },
    "MultiheadGAT": {
        "single_k": {
            "repo_id": "mojique/gat-single-k",
            "checkpoint": "gat_k60_best.pt",
            "config": "gat_k60_best.json",
            "revision": None,
        },
        "multi_k": {
            "repo_id": "mojique/gat-multi-k",
            "checkpoint": "gat_k60_k120_k300_k500_multi_best.pt",
            "config": "gat_k60_k120_k300_k500_multi_best.json",
            "revision": None,
        },
    },
}


class EvalModelWrapper(torch.nn.Module):
    """Wrap models to gracefully handle optional edge_index arguments."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        if edge_index is not None:
            try:
                return self.inner(x, edge_index)
            except TypeError:
                pass
        return self.inner(x)

SUMMARY_METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "f2",
    "nodes_mae",
    "nodes_rmse",
    "nodes_brier",
    "nodes_js",
    "nodes_hellinger",
    "nodes_r2",
    "nodes_pearson",
    "nodes_spearman",
    "img_mae_macro",
    "img_rmse_macro",
    "img_r2_macro",
    "img_smape_macro",
    "img_js",
    "img_emd1d",
    "img_mae_micro",
    "img_rmse_micro",
    "img_dice_macro",
]


class PyGMetricLoader:
    """Adapter to present PyG mini-batches to utils.metrics collectors."""

    def __init__(self, pyg_loader: PyGDataLoader) -> None:
        self._pyg_loader = pyg_loader

    def __iter__(self):
        for batch in self._pyg_loader:
            if hasattr(batch, "batch"):
                num_graphs = int(getattr(batch, "num_graphs", torch.max(batch.batch).item() + 1))
                nodes_per_graph = torch.bincount(batch.batch, minlength=num_graphs).tolist()
            else:
                nodes_per_graph = [batch.x.shape[0]]
            yield {
                "x": batch.x,
                "edge_index": batch.edge_index,
                "y": batch.y,
                "meta": {"nodes_per_graph": nodes_per_graph},
            }

    def __len__(self) -> int:
        return len(self._pyg_loader)


def infer_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)

    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")


def gather_checkpoint_paths(
    explicit_paths: List[str],
    directories: List[str],
    default_dir: Path,
) -> List[Path]:
    paths: List[Path] = []
    for p in explicit_paths:
        candidate = Path(p)
        if candidate.is_file():
            paths.append(candidate.resolve())
        else:
            raise FileNotFoundError(f"Checkpoint not found: {candidate}")

    dirs_to_search = [Path(d) for d in directories] if directories else [default_dir]
    exts = {".pt", ".pth"}
    for d in dirs_to_search:
        if not d.exists():
            continue
        for ckpt in sorted(d.glob("*")):
            if ckpt.suffix.lower() in exts and ckpt.is_file():
                paths.append(ckpt.resolve())

    unique_paths = sorted(dict.fromkeys(paths))
    if not unique_paths:
        raise RuntimeError("No checkpoints found for evaluation.")
    return unique_paths


def download_default_hf_checkpoints() -> List[Path]:
    """
    Download the default set of Hugging Face checkpoints specified in HF_MODEL_REGISTRY.
    Returns the local filesystem paths to the downloaded checkpoint files.
    """

    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - informative failure path
        raise ImportError(
            "huggingface_hub is required to download checkpoints from Hugging Face. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    downloaded: List[Path] = []
    for arch, variants in HF_MODEL_REGISTRY.items():
        for variant, spec in variants.items():
            repo_id = spec["repo_id"]
            checkpoint_filename = spec["checkpoint"]
            config_filename = spec["config"]
            revision = spec.get("revision")

            if repo_id is None or checkpoint_filename is None or config_filename is None:
                raise ValueError(
                    f"Incomplete Hugging Face spec for {arch} ({variant}). "
                    "Ensure repo_id, checkpoint, and config are populated."
                )

            ckpt_local = Path(
                hf_hub_download(repo_id=repo_id, filename=checkpoint_filename, revision=revision)
            )
            # Ensure the associated architecture metadata is also cached locally.
            hf_hub_download(repo_id=repo_id, filename=config_filename, revision=revision)
            downloaded.append(ckpt_local)
    return downloaded


def load_model_from_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    num_classes_eff: int,
) -> Tuple[torch.nn.Module, str, str, Dict]:
    meta_path = ckpt_path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing sidecar JSON for checkpoint: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    arch = meta.get("architecture", {}) or {}
    model_class_name = arch.get("model_class")
    if model_class_name not in MODEL_CLASS_MAP:
        raise ValueError(f"Unsupported model_class '{model_class_name}' in {meta_path}")

    model_cls = MODEL_CLASS_MAP[model_class_name]
    label = meta.get("train_config", {}).get("model_name") or ckpt_path.stem
    cfg = SimpleNamespace(ckpt_path=str(ckpt_path), device=device)
    model = model_cls.load_model(cfg, num_classes_eff=num_classes_eff)
    model.to(device)
    model.eval()
    wrapped = EvalModelWrapper(model)
    wrapped.to(device)
    wrapped.eval()
    return wrapped, model_class_name, label, meta


def build_datasets_and_loaders(
    seed: int,
    k_value: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[PyGDataLoader, PyGDataLoader, DeepGlobeDataset, Dict]:
    names, class_rgb_values, unknown_index = load_class_palette(const.CLASS_CSV)
    base_dataset = DeepGlobeDataset(const.TRAIN_DATA_DIR, class_rgb_values)

    n = len(base_dataset)
    if n == 0:
        raise RuntimeError("Training dataset is empty.")
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(0.8 * n)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    base_train = Subset(base_dataset, train_idx.tolist())
    base_test = Subset(base_dataset, test_idx.tolist())

    k_values = [int(k_value)]
    train_ds = SuperpixelGraphDatasetV2(
        base=base_train,
        class_rgb_values=class_rgb_values,
        k_values=k_values,
        unknown_index=unknown_index,
        normalize_targets=True,
        device=device,
        samples_per_image=1,
    )
    test_ds = SuperpixelGraphDatasetV2(
        base=base_test,
        class_rgb_values=class_rgb_values,
        k_values=k_values,
        unknown_index=unknown_index,
        normalize_targets=True,
        device=device,
        samples_per_image=1,
    )

    pin_memory = device.type == "cuda"
    train_loader = PyGDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=pin_memory,
    )
    test_loader = PyGDataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=pin_memory,
    )

    eff_names = [
        name
        for idx, name in enumerate(names)
        if unknown_index is None or idx != unknown_index
    ]

    meta = {
        "names": names,
        "effective_names": eff_names,
        "class_rgb_values": class_rgb_values,
        "unknown_index": unknown_index,
    }
    return train_loader, test_loader, base_dataset, meta


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    if t.numel() == 0:
        return np.empty((0,), dtype=np.float32)
    return t.detach().cpu().numpy()


def compute_metrics_for_model(
    model: torch.nn.Module,
    train_loader: PyGDataLoader,
    test_loader: PyGDataLoader,
) -> Tuple[Dict[str, float], Dict[str, Dict[int, float]], Dict[str, Dict[int, float]], torch.Tensor]:
    normalize_feats = False
    train_metrics_loader = PyGMetricLoader(train_loader)
    test_metrics_loader = PyGMetricLoader(test_loader)

    y_score_train, y_true_train = collect_image_scores(model, train_metrics_loader, normalize_feats)
    thresholds = calibrate_thresholds(y_score_train, y_true_train, beta=2.0)
    y_score_test, y_true_test = collect_image_scores(model, test_metrics_loader, normalize_feats)
    precision, recall, f1, f2 = example_based_metrics(y_true_test, y_score_test, thresholds)

    regression_data = collect_regression_data(model, test_metrics_loader, normalize_feats)
    node_pred = tensor_to_numpy(regression_data["nodes"]["y_pred"])
    node_true = tensor_to_numpy(regression_data["nodes"]["y_true"])
    node_weights = tensor_to_numpy(regression_data["nodes"]["weights"])
    img_pred = tensor_to_numpy(regression_data["images"]["y_pred"])
    img_true = tensor_to_numpy(regression_data["images"]["y_true"])
    img_weights = tensor_to_numpy(regression_data["images"]["weights"])

    metrics_summary: Dict[str, float] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "nodes_mae": float("nan"),
        "nodes_rmse": float("nan"),
        "nodes_brier": float("nan"),
        "nodes_js": float("nan"),
        "nodes_hellinger": float("nan"),
        "nodes_r2": float("nan"),
        "nodes_pearson": float("nan"),
        "nodes_spearman": float("nan"),
        "img_mae_macro": float("nan"),
        "img_rmse_macro": float("nan"),
        "img_r2_macro": float("nan"),
        "img_smape_macro": float("nan"),
        "img_js": float("nan"),
        "img_emd1d": float("nan"),
        "img_mae_micro": float("nan"),
        "img_rmse_micro": float("nan"),
        "img_dice_macro": float("nan"),
    }

    per_class_nodes: Dict[str, Dict[int, float]] = {}
    per_class_images: Dict[str, Dict[int, float]] = {}

    if node_pred.size > 0 and node_true.size > 0:
        metrics_summary["nodes_mae"] = mae_weighted_nodes(node_true, node_pred, node_weights if node_weights.size else None)
        metrics_summary["nodes_rmse"] = rmse_weighted_nodes(node_true, node_pred, node_weights if node_weights.size else None)
        metrics_summary["nodes_brier"] = brier_score_weighted_nodes(node_true, node_pred, node_weights if node_weights.size else None)
        metrics_summary["nodes_js"] = _js_divergence_batch(node_true, node_pred, weights=node_weights if node_weights.size else None)
        metrics_summary["nodes_hellinger"] = _hellinger_batch(node_true, node_pred, weights=node_weights if node_weights.size else None)

        r2_per_class, r2_macro = r2_per_class_nodes(node_true, node_pred, node_weights if node_weights.size else None)
        pearson_per_class, spearman_per_class, pearson_macro, spearman_macro = pearson_spearman_per_class_nodes(node_true, node_pred)

        metrics_summary["nodes_r2"] = r2_macro
        metrics_summary["nodes_pearson"] = pearson_macro
        metrics_summary["nodes_spearman"] = spearman_macro

        per_class_nodes["r2"] = r2_per_class
        per_class_nodes["pearson"] = pearson_per_class
        per_class_nodes["spearman"] = spearman_per_class

    if img_pred.size > 0 and img_true.size > 0:
        metrics_summary["img_js"] = js_divergence_images(img_true, img_pred)
        metrics_summary["img_emd1d"] = emd_1d_images(img_true, img_pred)
        scores = per_class_regression_scores_images(img_true, img_pred, img_weights if img_weights.size else None)
        metrics_summary["img_mae_macro"] = scores["macro"]["mae"]
        metrics_summary["img_rmse_macro"] = scores["macro"]["rmse"]
        metrics_summary["img_r2_macro"] = scores["macro"]["r2"]
        metrics_summary["img_smape_macro"] = scores["macro"]["smape"]
        metrics_summary["img_mae_micro"] = scores["micro"]["mae"]
        metrics_summary["img_rmse_micro"] = scores["micro"]["rmse"]

        dice_per_class, dice_macro = soft_dice_per_class(img_true, img_pred)
        metrics_summary["img_dice_macro"] = dice_macro
        per_class_images["dice"] = dice_per_class

        per_class_images["mae"] = {int(c): vals["mae"] for c, vals in scores["per_class"].items()}
        per_class_images["rmse"] = {int(c): vals["rmse"] for c, vals in scores["per_class"].items()}
        per_class_images["r2"] = {int(c): vals["r2"] for c, vals in scores["per_class"].items()}
        per_class_images["smape"] = {int(c): vals["smape"] for c, vals in scores["per_class"].items()}

    return metrics_summary, per_class_nodes, per_class_images, thresholds


def summarize_per_class(
    model_label: str,
    model_class: str,
    ckpt_path: Path,
    k_value: int,
    seed: int,
    per_class_nodes: Dict[str, Dict[int, float]],
    per_class_images: Dict[str, Dict[int, float]],
) -> Tuple[List[Dict], List[Dict]]:
    node_rows: List[Dict] = []
    image_rows: List[Dict] = []

    node_keys = sorted(per_class_nodes.keys())
    node_classes = set()
    for key in node_keys:
        node_classes.update(per_class_nodes[key].keys())

    for class_idx in sorted(node_classes):
        row = {
            "model_label": model_label,
            "model_class": model_class,
            "ckpt_path": str(ckpt_path),
            "k": int(k_value),
            "seed": int(seed),
            "class_index": int(class_idx),
        }
        for key in node_keys:
            row[f"nodes_{key}"] = per_class_nodes[key].get(class_idx, float("nan"))
        node_rows.append(row)

    image_keys = sorted(per_class_images.keys())
    image_classes = set()
    for key in image_keys:
        image_classes.update(per_class_images[key].keys())

    for class_idx in sorted(image_classes):
        row = {
            "model_label": model_label,
            "model_class": model_class,
            "ckpt_path": str(ckpt_path),
            "k": int(k_value),
            "seed": int(seed),
            "class_index": int(class_idx),
        }
        for key in image_keys:
            row[f"img_{key}"] = per_class_images[key].get(class_idx, float("nan"))
        image_rows.append(row)

    return node_rows, image_rows


def plot_metrics(
    summary_df: pd.DataFrame,
    output_dir: Path,
    k_value: int,
    seed: Optional[int],
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    filtered = summary_df[summary_df["k"] == k_value]
    if seed is not None and "seed" in filtered.columns:
        filtered = filtered[filtered["seed"] == seed]
    if filtered.empty:
        raise RuntimeError(f"No rows available for plotting with k={k_value} and seed={seed}.")

    filtered = filtered.sort_values("model_label")

    for metric in SUMMARY_METRIC_COLUMNS:
        if metric not in filtered.columns:
            continue
        series = filtered[metric].astype(float)
        if series.isna().all():
            continue

        fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * len(filtered))))
        ax.barh(filtered["model_label"], series)
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_ylabel("Model")
        title = f"{metric.replace('_', ' ').title()} (k={k_value}"
        if seed is not None:
            title += f", seed={seed}"
        title += ")"
        ax.set_title(title)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()
        out_path = plots_dir / f"{metric}_k{k_value}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def save_thresholds(
    thresholds: torch.Tensor,
    class_names: List[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thresholds": thresholds.tolist(),
        "class_names": class_names,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def visualize_masks(
    models_info: List[Tuple[str, torch.nn.Module]],
    dataset: DeepGlobeDataset,
    image_rel_path: str,
    k_value: int,
    class_rgb_values: List[Tuple[int, int, int]],
    unknown_index: Optional[int],
    output_dir: Path,
    device: torch.device,
) -> None:
    rel_path = Path(image_rel_path)
    base_dir = Path(const.TRAIN_DATA_DIR)
    img_path = (base_dir / rel_path).resolve()

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    if img_path.suffix == "":
        raise RuntimeError(f"Provided path lacks extension: {image_rel_path}")

    if "_sat" not in img_path.stem:
        raise RuntimeError("Expected image filename to contain '_sat'.")

    mask_stem = img_path.stem.replace("_sat", "_mask")
    mask_path = img_path.with_name(mask_stem).with_suffix(".png")

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found for visualization: {mask_path}")

    target_idx = None
    for idx, path in enumerate(dataset.image_paths):
        if Path(path) == img_path:
            target_idx = idx
            break
    if target_idx is None:
        raise RuntimeError(f"Image {img_path} not part of the dataset.")

    img_t, img_rgb, mask_t = dataset[target_idx]
    from feature_extractor import extract_image_feature_map, get_slic_graph  # local import

    with torch.inference_mode():
        feature_map = extract_image_feature_map(img_t, device=device)
        X, edge_index, sp = get_slic_graph(feature_map, img_rgb, k=k_value, device=device)

    mask_true = mask_t.numpy()
    palette = np.array(class_rgb_values, dtype=np.uint8)
    mask_true_rgb = palette[mask_true]

    fig, axes = plt.subplots(1, len(models_info) + 1, figsize=(4 * (len(models_info) + 1), 4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(mask_true_rgb)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    for ax_idx, (label, model) in enumerate(models_info, start=1):
        with torch.inference_mode():
            logits = model(X.to(device), edge_index.to(device))
            prob = torch.softmax(logits, dim=1)
        pred_nodes = torch.argmax(prob, dim=1).cpu().numpy()
        pred_mask = np.zeros_like(sp, dtype=np.int32)
        for sp_id, cls_eff in enumerate(pred_nodes):
            if unknown_index is None or cls_eff < unknown_index:
                actual_cls = int(cls_eff)
            else:
                actual_cls = int(cls_eff + 1)
            pred_mask[sp == sp_id] = actual_cls
        pred_rgb = palette[pred_mask]
        axes[ax_idx].imshow(pred_rgb)
        axes[ax_idx].set_title(label)
        axes[ax_idx].axis("off")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"viz_{rel_path.stem}_k{k_value}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate graph-based models and plot metrics.")
    parser.add_argument("--k", type=int, required=True, help="Superpixel K value to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument("--ckpt", action="append", default=[], help="Specific checkpoint file (.pt/.pth). Repeatable.")
    parser.add_argument("--ckpt-dir", action="append", default=[], help="Directory containing checkpoints. Repeatable.")
    parser.add_argument(
        "--use-hf-models",
        action="store_true",
        help="Download and include the default Hugging Face checkpoints defined in this file.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader workers.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda, mps).")
    parser.add_argument("--results-csv", type=str, help="When provided, load metrics from CSV and only plot results.")
    parser.add_argument("--plot-per-class", action="store_true", help="Generate per-class plots (only valid when computing metrics).")
    parser.add_argument("--viz-image", type=str, help="Relative path under train data for qualitative mask visualization.")
    args = parser.parse_args()

    device = infer_device(args.device)
    print(f"Using device: {device}")
    output_root = const.ARTIFACTS_DIR / "eval"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.results_csv:
        summary_path = Path(args.results_csv)
        if not summary_path.exists():
            raise FileNotFoundError(f"Results CSV not found: {summary_path}")
        summary_df = pd.read_csv(summary_path)
        plot_metrics(summary_df, output_root, args.k, args.seed)
        if args.plot_per_class:
            print("Per-class plotting is only supported when metrics are computed in this run.")
        if args.viz_image:
            print("Visualization skipped because --results-csv was provided.")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, test_loader, base_dataset, meta = build_datasets_and_loaders(
        seed=args.seed,
        k_value=args.k,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    class_names_eff = meta["effective_names"]
    class_rgb_values = meta["class_rgb_values"]
    unknown_index = meta["unknown_index"]
    num_classes_eff = len(class_rgb_values) - (1 if unknown_index is not None and 0 <= unknown_index < len(class_rgb_values) else 0)

    ckpt_paths: List[Path] = []
    local_error: Optional[RuntimeError] = None
    try:
        ckpt_paths = gather_checkpoint_paths(args.ckpt, args.ckpt_dir, output_root)
    except RuntimeError as err:
        local_error = err
        ckpt_paths = []

    hf_ckpt_paths: List[Path] = []
    if args.use_hf_models:
        hf_ckpt_paths = download_default_hf_checkpoints()

    if not ckpt_paths and not args.use_hf_models:
        # Preserve previous behaviour when Hugging Face models are not requested.
        if local_error is not None:
            raise local_error
    ckpt_paths.extend(hf_ckpt_paths)
    if not ckpt_paths:
        raise RuntimeError(
            "No checkpoints found locally or on Hugging Face. "
            "Provide --ckpt/--ckpt-dir or enable --use-hf-models."
        )
    # Deduplicate while preserving order.
    ckpt_paths = list(dict.fromkeys(ckpt_paths))

    summary_rows: List[Dict] = []
    per_class_node_rows: List[Dict] = []
    per_class_image_rows: List[Dict] = []
    thresholds_dir = output_root / "thresholds"
    viz_models_info: List[Tuple[str, torch.nn.Module]] = []

    for ckpt_path in tqdm(ckpt_paths, desc="Evaluating checkpoints"):
        model, model_class, label, meta_json = load_model_from_checkpoint(ckpt_path, device, num_classes_eff)
        metrics_summary, per_class_nodes, per_class_images, thresholds = compute_metrics_for_model(
            model, train_loader, test_loader
        )

        summary_row = {
            "model_label": label,
            "model_class": model_class,
            "ckpt_path": str(ckpt_path),
            "k": int(args.k),
            "seed": int(args.seed),
        }
        summary_row.update(metrics_summary)
        summary_rows.append(summary_row)

        node_rows, image_rows = summarize_per_class(label, model_class, ckpt_path, args.k, args.seed, per_class_nodes, per_class_images)
        per_class_node_rows.extend(node_rows)
        per_class_image_rows.extend(image_rows)

        threshold_path = thresholds_dir / f"{ckpt_path.stem}_thresholds.json"
        save_thresholds(thresholds, class_names_eff[: len(thresholds)], threshold_path)

        if args.viz_image:
            viz_models_info.append((label, model))

        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_root / f"metrics_summary_k{args.k}_seed{args.seed}.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    per_class_dir = output_root / "per_class"
    per_class_dir.mkdir(parents=True, exist_ok=True)
    if per_class_node_rows:
        pd.DataFrame(per_class_node_rows).to_csv(per_class_dir / f"nodes_k{args.k}_seed{args.seed}.csv", index=False)
    if per_class_image_rows:
        pd.DataFrame(per_class_image_rows).to_csv(per_class_dir / f"images_k{args.k}_seed{args.seed}.csv", index=False)

    plot_metrics(summary_df, output_root, args.k, args.seed)

    if args.plot_per_class:
        print("Per-class plots not yet implemented; see per-class CSV outputs.")

    if args.viz_image:
        visualize_masks(
            models_info=viz_models_info,
            dataset=base_dataset,
            image_rel_path=args.viz_image,
            k_value=args.k,
            class_rgb_values=class_rgb_values,
            unknown_index=unknown_index,
            output_dir=output_root / "viz",
            device=device,
        )


if __name__ == "__main__":
    main()

