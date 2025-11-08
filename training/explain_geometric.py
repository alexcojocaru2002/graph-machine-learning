import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ExplanationType, MaskType

from torch.utils.data import Subset

import const
from config.geometric_train_config import GeometricTrainConfig
from datasets.superpixel_graph_dataset_v2 import SuperpixelGraphDatasetV2
from dataset_loader import DeepGlobeDataset
from load_palette import load_class_palette
from training.train_geometric import get_device


def load_model_from_json(meta_path: Path) -> Tuple[torch.nn.Module, dict]:
    from models.gcn import GCN2
    from models.gat_geometric import MultiheadGAT

    with open(meta_path, "r") as f:
        meta = json.load(f)

    arch = meta["architecture"]
    model_class = arch.pop("model_class")

    model_registry = {
        "GCN2": GCN2,
        "MultiheadGAT": MultiheadGAT,
    }

    if model_class not in model_registry:
        raise ValueError(f"Unknown model class: {model_class}")

    model = model_registry[model_class](**arch)
    return model, meta


def geometric_explainer_entrypoint(
    config: GeometricTrainConfig,
    model: Optional[torch.nn.Module] = None,
    val_image_limit: int = 1,
    node_idx: Optional[int] = None,   # None = explain all nodes
    explain_epochs: int = 200,
    target_superpixel_idx: Optional[int] = None,  # NEW: single-node explanation
):
    """
    Generates a heatmap of node importance for the first validation image.
    Optionally generates a second heatmap for a single target superpixel.
    """

    device = get_device()

    # Load model and metadata
    meta_path = const.ARTIFACTS_DIR / f"{config.model_name}_best.json"
    ckpt_weights = const.ARTIFACTS_DIR / f"{config.model_name}_best_weights.pt"

    if not meta_path.exists() or not ckpt_weights.exists():
        raise FileNotFoundError("Missing JSON sidecar or weights for model.")

    if model is None:
        model, meta = load_model_from_json(meta_path)
    else:
        with open(meta_path, "r") as f:
            meta = json.load(f)

    model = model.to(device)
    model.load_state_dict(torch.load(ckpt_weights, map_location=device))
    model.eval()
    print(f"[INFO] Loaded model {config.model_name} with weights.")

    # Load dataset
    _, class_rgb_values, unknown_index = load_class_palette(const.CLASS_CSV)
    val_image_ids = meta.get("val_image_ids", [])
    if not val_image_ids:
        raise ValueError("No validation image IDs found in sidecar JSON.")

    val_ids_subset = val_image_ids[:val_image_limit]

    base = DeepGlobeDataset(const.TRAIN_DATA_DIR, class_rgb_values)
    subset_indices = [
        i for i, p in enumerate(base.image_paths)
        if Path(p).stem in val_ids_subset
    ]
    base_val = Subset(base, subset_indices)

    val_ds = SuperpixelGraphDatasetV2(
        base=base_val,
        class_rgb_values=class_rgb_values,
        k_values=config.k_values,
        normalize_targets=True,
        device=device,
        unknown_index=unknown_index,
    )

    # Pick first image
    data = val_ds[0].to(device)
    val_image_id = val_ids_subset[0]
    print(f"[INFO] Explaining graph from validation image: {val_image_id}")

    # Node-level explanation
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=explain_epochs),
        explanation_type=ExplanationType.model,
        node_mask_type=MaskType.object,   # Node-level importance
        edge_mask_type=None,               # Skip edge importance
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="log_probs",
        )
    )

    # --- 1) General graph explanation ---
    explanation = explainer(x=data.x, edge_index=data.edge_index, index=node_idx)
    node_importance = explanation.node_mask.detach().cpu().numpy()  # [N]

    sp_map = val_ds._load_graph(0, data.x, data.meta.get("img_rgb"), k=config.k_values[0])[2]
    img_rgb = data.meta.get("img_rgb")
    if img_rgb is None:
        img_rgb = val_ds.base[0][1]

    heatmap = np.zeros_like(sp_map, dtype=float)
    for i, score in enumerate(node_importance):
        heatmap[sp_map == i] = score
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title(f"Node Importance Heatmap: {val_image_id}")
    plt.show()

    # --- 2) Single-target superpixel explanation ---
    target_superpixel_idx = 0
    target_heatmap = None
    target_node_importance = None
    if target_superpixel_idx is not None:
        print(f"[INFO] Explaining target superpixel {target_superpixel_idx}")
        target_explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            index=target_superpixel_idx
        )
        target_node_importance = target_explanation.node_mask.detach().cpu().numpy()

        # Map importance to pixels
        target_heatmap = np.zeros_like(sp_map, dtype=float)
        for i, score in enumerate(target_node_importance):
            target_heatmap[sp_map == i] = score

        # Normalize all superpixels except the target superpixel
        mask = sp_map != target_superpixel_idx
        if np.any(mask):
            min_val, max_val = target_heatmap[mask].min(), target_heatmap[mask].max()
            target_heatmap[mask] = (target_heatmap[mask] - min_val) / (max_val - min_val + 1e-8)

        # Separate mask for target superpixel
        target_mask = sp_map == target_superpixel_idx

        # Plot original image
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)

        # Overlay normalized heatmap for other superpixels
        plt.imshow(target_heatmap, cmap='jet', alpha=0.5)

        # Overlay target superpixel as white
        green_overlay = np.zeros((*sp_map.shape, 4), dtype=float)  # RGBA
        green_overlay[target_mask] = [0.0, 1.0, 0.0, 1.0]  # neon green with full alpha
        plt.imshow(green_overlay)

        plt.axis('off')
        plt.title(f"Target Superpixel {target_superpixel_idx} Influence Heatmap: {val_image_id}")
        plt.show()

        return heatmap, node_importance, target_heatmap, target_node_importance, val_image_id
