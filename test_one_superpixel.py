import argparse
from pathlib import Path

import torch
from torchvision.models import vgg16, VGG16_Weights

from eval import (
    EvalConfig,
    make_dataset,
    load_model,
    _plot_random_superpixel_probabilities,
)

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

    # Build backbone (same as eval.main)
    backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(cfg.feature_device).eval()
    img_size = None
    if cfg.img_size_w is not None and cfg.img_size_h is not None:
        img_size = (int(cfg.img_size_w), int(cfg.img_size_h))

    # Build dataset (uses cfg.train_dir by default)
    ds_full, names, class_rgb_values, unknown_index = make_dataset(
        cfg.test_dir,
        cfg.class_csv,
        img_size,
        cfg.k_values,
        cfg.feature_device,
        cfg.cache_dir,
        normalize_targets=False,
        backbone=backbone,
        device=cfg.device,
        hsv_threshold=cfg.hsv_threshold,
    )

    # choose k
    k_for_plot = args.k if args.k is not None else int(ds_full.k_values[0])
    if k_for_plot not in ds_full.k_values:
        print(f"[test] requested k={k_for_plot} not in dataset k_values={ds_full.k_values}; using {ds_full.k_values[0]}")
        k_for_plot = int(ds_full.k_values[0])

    # load model
    num_classes_eff = len(class_rgb_values) - (1 if (unknown_index is not None and 0 <= unknown_index < len(class_rgb_values)) else 0)
    model = load_model(cfg, num_classes_eff)

    # call plotting helper for one random superpixel
    _plot_random_superpixel_probabilities(
        ds_full=ds_full,
        model_list=[model],
        model_labels=[Path(cfg.ckpt_path).stem],
        names=names,
        class_rgb_values=class_rgb_values,
        unknown_index=unknown_index,
        k_value=k_for_plot,
        normalize_node_features=cfg.normalize_node_features,
        device=next(model.parameters()).device,
        save_dir=cfg.save_plots_dir,
        show=True,
        image_index=args.image_index,
    )
    print(f"[test] plot saved to {cfg.save_plots_dir}")

if __name__ == "__main__":
    main()