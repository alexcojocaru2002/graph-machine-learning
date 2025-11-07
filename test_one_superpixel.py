import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd
from pathlib import Path
import random

import const
from dataset_loader import DeepGlobeDataset
from datasets.superpixel_graph_dataset_v2 import SuperpixelGraphDatasetV2
from models.gcn import GCN2
from models.gat_geometric import MultiheadGAT
from skimage.segmentation import find_boundaries


def load_class_info(csv_path):
    """Load class names and RGB values from CSV."""
    df = pd.read_csv(csv_path)
    class_names = df['name'].tolist()
    class_rgb_values = [(int(r), int(g), int(b)) for r, g, b in zip(df['r'], df['g'], df['b'])]
    return class_names, class_rgb_values


def create_border_overlay(image, borders, color=(255, 0, 0), thickness=1):
    """Overlay borders on an image."""
    overlay = image.copy()
    overlay[borders] = color
    return overlay


def visualize_superpixels(img_rgb, sp):
    """Create an image with superpixel borders."""
    borders = find_boundaries(sp, mode='thick')
    return create_border_overlay(img_rgb, borders)


def mask_to_rgb(mask, class_rgb_values):
    """Convert class mask to RGB visualization."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in enumerate(class_rgb_values):
        rgb[mask == class_idx] = color
    return rgb


def get_superpixel_predictions(model, data, device):
    """Get per-superpixel area fraction predictions from the model."""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        preds = model(data.x, data.edge_index)  # [N, num_classes] area fractions
        pred_probs = torch.softmax(preds, dim=-1)  # convert logits → probabilities
    return pred_probs.cpu().numpy()


def calculate_mse_per_superpixel(pred, gt):
    """Calculate MSE loss per superpixel.
    
    Args:
        pred: [N, num_classes] predicted area fractions
        gt: [N, num_classes] ground truth area fractions
    
    Returns:
        mse_per_sp: [N] MSE for each superpixel
    """
    mse_per_sp = np.mean((pred - gt) ** 2, axis=1)  # [N]
    return mse_per_sp


def highlight_superpixel(sp, sp_id, color=(255, 0, 0), alpha=0.5):
    """Create a mask highlighting a specific superpixel."""
    mask = (sp == sp_id)
    return mask


def main():
    # Config
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    ckpt_path = "D:/Work/graph-machine-learning/artifacts/gat_k60_best.ckpt"
    k_value = 60
    random_seed = 42
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Load class info
    class_names, class_rgb_values = load_class_info(const.CLASS_CSV)
    num_classes_total = len(class_rgb_values)
    
    # Exclude 'unknown' class (last class) from predictions and evaluation
    class_names_no_unknown = [name for name in class_names if name.lower() != 'unknown']
    num_classes = len(class_names_no_unknown)
    
    print(f"Loaded {num_classes_total} classes: {class_names}")
    print(f"Using {num_classes} classes for model (excluding 'unknown')")
    
    # Load dataset
    base_dataset = DeepGlobeDataset(
        data_dir=const.TRAIN_DATA_DIR,
        class_rgb_values=class_rgb_values,
        img_size=const.IMG_SIZE
    )
    
    dataset = SuperpixelGraphDatasetV2(
        base=base_dataset,
        class_rgb_values=class_rgb_values,
        k_values=[k_value],
        unknown_index=None,  # None means unknown is excluded from targets
        normalize_targets=True,
        device=device
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Pick a random sample
    random_idx = random.randint(0, len(dataset) - 1)
    data = dataset[random_idx]
    image_idx = data.meta["image_idx"]
    k = data.meta["k"]
    
    print(f"\nSelected random sample {random_idx}")
    print(f"  Image index: {image_idx}, K: {k}")
    print(f"  Num superpixels: {data.meta['num_nodes']}")
    
    # Load the original image, RGB, and mask
    img_t, img_rgb, mask_t = base_dataset[image_idx]
    mask = mask_t.numpy()
    
    # Load the superpixel segmentation (from cache)
    key = Path(base_dataset.image_paths[image_idx]).stem
    sp_path = dataset.graph_dir / f"{key}_k{k}_sp.npy"
    sp = np.load(sp_path)
    
    # Load model
    print(f"\nLoading model from {ckpt_path}...")
    
    # Create a simple config object for loading
    class Config:
        pass
    cfg = Config()
    cfg.ckpt_path = ckpt_path
    cfg.device = device
    
    model = MultiheadGAT.load_model(cfg, num_classes_eff=num_classes)
    model = model.to(device)
    print(f"Model loaded successfully! Output classes: {num_classes} (unknown excluded)")
    
    # Get predictions (area fractions) - unknown class excluded
    pred_fractions = get_superpixel_predictions(model, data, device)  # [N, num_classes]
    gt_fractions_raw = data.y.cpu().numpy()  # [N, num_classes_total] includes unknown
    
    # Remove unknown class (last column) from ground truth
    gt_fractions = gt_fractions_raw[:, :-1]  # [N, num_classes] exclude last column
    
    # Verify shapes match (should be [N, num_classes] without unknown)
    assert pred_fractions.shape[1] == num_classes, f"Prediction shape mismatch: {pred_fractions.shape[1]} != {num_classes}"
    assert gt_fractions.shape[1] == num_classes, f"Ground truth shape mismatch: {gt_fractions.shape[1]} != {num_classes}"
    
    print(f"\nPredictions shape: {pred_fractions.shape} (unknown class excluded)")
    print(f"Ground truth shape: {gt_fractions.shape} (unknown class excluded)")
    
    # Calculate MSE per superpixel
    mse_per_sp = calculate_mse_per_superpixel(pred_fractions, gt_fractions)
    mean_mse = np.mean(mse_per_sp)
    
    print(f"\nMean MSE across all superpixels: {mean_mse:.6f}")
    print(f"MSE range: [{mse_per_sp.min():.6f}, {mse_per_sp.max():.6f}]")
    
    # Select interesting superpixels: best, worst, and median MSE
    sorted_indices = np.argsort(mse_per_sp)
    best_sp_idx = sorted_indices[0]
    worst_sp_idx = sorted_indices[-1]
    median_sp_idx = sorted_indices[len(sorted_indices) // 2]
    
    selected_sps = [
        ("Best (Lowest MSE)", best_sp_idx, mse_per_sp[best_sp_idx], 'green'),
        ("Median MSE", median_sp_idx, mse_per_sp[median_sp_idx], 'yellow'),

        ("Worst (Highest MSE)", worst_sp_idx, mse_per_sp[worst_sp_idx], 'red'),
    ]
    
    print(f"\nSelected superpixels for visualization:")
    for name, idx, mse_val, _ in selected_sps:
        print(f"  {name}: SP#{idx}, MSE={mse_val:.6f}")
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: Image visualizations
    ax_img = fig.add_subplot(gs[0, :])
    
    # Create image with highlighted superpixels
    img_display = img_rgb.copy().astype(float) / 255.0
    
    # Overlay each selected superpixel
    colors_map = {'green': (0, 1, 0), 'yellow': (1, 1, 0), 'red': (1, 0, 0)}
    for i, (name, sp_idx, mse_val, color) in enumerate(selected_sps):
        mask = (sp == sp_idx)
        overlay_color = colors_map[color]
        for c in range(3):
            img_display[:, :, c] = np.where(mask, 
                                           img_display[:, :, c] * 0.4 + overlay_color[c] * 0.6,
                                           img_display[:, :, c])
        
        # Add text label
        coords = np.argwhere(mask)
        if len(coords) > 0:
            cy, cx = coords.mean(axis=0).astype(int)
            ax_img.text(cx, cy, str(i+1), color='white', fontsize=16, 
                       fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='circle', facecolor=color, alpha=0.8))
    
    # Draw superpixel borders
    borders = find_boundaries(sp, mode='thick')
    img_display[borders] = [1, 1, 1]  # white borders
    
    ax_img.imshow(img_display)
    ax_img.set_title(f"Highlighted Superpixels (K={k}, Total={data.meta['num_nodes']}, Mean MSE={mean_mse:.6f})",
                    fontsize=14, fontweight='bold')
    ax_img.axis('off')
    
    # Rows 2-3: Bar charts for each selected superpixel
    for i, (name, sp_idx, mse_val, color) in enumerate(selected_sps):
        ax = fig.add_subplot(gs[(i // 2) + 1, (i % 2) * 2: (i % 2) * 2 + 2])
        
        x = np.arange(num_classes)  # All classes (unknown already excluded)
        width = 0.35
        
        pred = pred_fractions[sp_idx]
        gt = gt_fractions[sp_idx]
        
        bars1 = ax.bar(x - width/2, gt, width, label='Ground Truth', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, pred, width, label='Predicted', alpha=0.8, color='coral')
        
        ax.set_xlabel('Class (Unknown Excluded)', fontsize=11)
        ax.set_ylabel('Area Fraction', fontsize=11)
        ax.set_title(f"{i+1}. {name} (SP#{sp_idx}, MSE={mse_val:.6f})",
                    fontsize=12, fontweight='bold', color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names_no_unknown, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, max(1.0, gt.max() * 1.1, pred.max() * 1.1)])
    
    plt.tight_layout()
    
    # Save the figure
    output_path = Path("test_one_superpixel_result.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_path}")
    
    plt.show()
    
    # Print detailed statistics
    print(f"\n" + "="*60)
    print(f"REGRESSION METRICS (MSE Loss) - Unknown Class Excluded")
    print("="*60)
    print(f"Mean MSE: {mean_mse:.6f}")
    print(f"Std MSE:  {np.std(mse_per_sp):.6f}")
    print(f"Min MSE:  {mse_per_sp.min():.6f}")
    print(f"Max MSE:  {mse_per_sp.max():.6f}")
    print(f"\nDetailed breakdown for selected superpixels:")
    print(f"Classes: {class_names_no_unknown}")
    for i, (name, sp_idx, mse_val, _) in enumerate(selected_sps):
        print(f"\n{i+1}. {name} (SP#{sp_idx}):")
        print(f"   MSE: {mse_val:.6f}")
        print(f"   GT fractions:   {[f'{v:.4f}' for v in gt_fractions[sp_idx]]}")
        print(f"   Pred fractions: {[f'{v:.4f}' for v in pred_fractions[sp_idx]]}")


if __name__ == "__main__":
    main()
