import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from config.eval_config import EvalConfig
from load_palette import load_class_palette
from datasets.superpixel_graph_dataset_v2 import SuperpixelGraphDatasetV2
from dataset_loader import DeepGlobeDataset
from models.gat import SPNodeRegressor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--k", type=int, default=None, help="SLIC k to visualize (must be in dataset k_values)")
    p.add_argument("--image_index", type=int, default=None, help="Optional image index to visualize")
    p.add_argument("--save_dir", type=str, default="artifacts/plots", help="Where to save plot")
    p.add_argument("--no_show", action="store_true", help="Do not show the plot interactively")
    args = p.parse_args()
    cfg = EvalConfig()
    cfg.ckpt_path = args.ckpt
    cfg.save_plots_dir = args.save_dir
    cfg.no_show = args.no_show

    # Palette and base dataset
    names, class_rgb_values, unknown_index = load_class_palette(cfg.class_csv)
    base = DeepGlobeDataset(cfg.train_dir, class_rgb_values, img_size=(cfg.img_size_w, cfg.img_size_h) if (cfg.img_size_w and cfg.img_size_h) else None)

    # Build SuperpixelGraphDatasetV2
    ds = SuperpixelGraphDatasetV2(
        base=base,
        class_rgb_values=class_rgb_values,
        k_values=list(cfg.k_values),
        unknown_index=unknown_index,
        normalize_targets=False,
        device=cfg.feature_device,
    )
    print("loaded dataset")
    # choose (image_idx, k)
    k_for_plot = args.k if args.k is not None else int(ds.k_values[0])
    if k_for_plot not in ds.k_values:
        print(f"[test] requested k={k_for_plot} not in dataset k_values={ds.k_values}; using {ds.k_values[0]}")
        k_for_plot = int(ds.k_values[0])
    # pick image index
    img_idx = args.image_index if args.image_index is not None else 0
    img_idx = int(max(0, min(len(base) - 1, img_idx)))
    # find linear idx
    lin_idx = None
    for i, (im_i, k) in enumerate(ds.index_map):
        if int(im_i) == img_idx and int(k) == int(k_for_plot):
            lin_idx = i
            break
    if lin_idx is None:
        print(f"[test] could not find sample for image_idx={img_idx} and k={k_for_plot}")
        return

    # Load graph sample and original image/mask
    data = ds[lin_idx]
    image_idx = int(data.meta["image_idx"]) if isinstance(data.meta, dict) else img_idx
    k_val = int(data.meta["k"]) if isinstance(data.meta, dict) else k_for_plot
    img_t, img_rgb, mask_t = base[image_idx]

    # Load superpixel map from ds cache
    stem = Path(base.image_paths[image_idx]).stem
    sp = np.load(ds.graph_dir / f"{stem}_k{k_val}_sp.npy")

    # Build and load model
    num_classes_eff = len(class_rgb_values) - (1 if (unknown_index is not None and 0 <= unknown_index < len(class_rgb_values)) else 0)
    model = SPNodeRegressor.load_model(cfg, num_classes_eff)
    device = next(model.parameters()).device

    # Forward pass
    x = data.x.to(device)
    if cfg.normalize_node_features:
        x = F.normalize(x, p=2, dim=1)
    edge_index = data.edge_index.to(device)
    with torch.inference_mode():
        logits = model(x, edge_index)
        pred_cls = torch.argmax(logits, dim=-1).detach().cpu().numpy()


    # Helper: palette without unknown
    eff_palette = [c for i, c in enumerate(class_rgb_values) if not (unknown_index is not None and i == unknown_index)]

    # Render predicted map
    H, W = sp.shape
    pred_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for node_id in range(pred_cls.shape[0]):
        c = int(pred_cls[node_id])
        color = eff_palette[c] if (0 <= c < len(eff_palette)) else (0, 0, 0)
        pred_rgb[sp == node_id] = color

    # Render ground-truth RGB
    mask_idx = mask_t.numpy().astype(np.int64)
    gt_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, (r, g, b) in enumerate(class_rgb_values):
        gt_rgb[mask_idx == idx] = (r, g, b)

    # Plot: image + boundaries, GT mask + boundaries, Pred + boundaries
    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot(1, 3, 1)
    ax.imshow(mark_boundaries(img_rgb, sp, color=(1, 0, 0)))
    ax.set_title(f"Image (k={k_val})")
    ax.axis('off')

    ax = plt.subplot(1, 3, 2)
    ax.imshow(mark_boundaries(gt_rgb, sp, color=(1, 0, 0)))
    ax.set_title("Mask + boundaries (GT)")
    ax.axis('off')

    ax = plt.subplot(1, 3, 3)
    ax.imshow(mark_boundaries(pred_rgb, sp, color=(1, 0, 0)))
    ax.set_title("Prediction + boundaries")
    ax.axis('off')

    Path(cfg.save_plots_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(cfg.save_plots_dir) / f"one_sp_image{image_idx}_k{k_val}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if not cfg.no_show:
        plt.show()
    plt.close(fig)
    print(f"[test] plot saved to {out_path}")

if __name__ == "__main__":
    main()