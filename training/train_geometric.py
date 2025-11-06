import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader
from pathlib import Path
import json
import pandas as pd

import const
from config.geometric_train_config import GeometricTrainConfig
from dataset_loader import DeepGlobeDataset
from datasets.superpixel_graph_dataset_v2 import SuperpixelGraphDatasetV2
from load_palette import load_class_palette

def geometric_training_entrypoint(model: torch.nn.Module, config: GeometricTrainConfig):
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    device = get_device()
    print(f"Using device: {device}")

    (train_ds, train_loader), (val_ds, val_loader), val_ids = create_split_pyg_loaders(
        config, device, train_ratio=0.8
    )

    model = model.to(device)

    _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        val_image_ids=val_ids,
    )

    print("Done.")

def test_model(model: torch.nn.Module, config: GeometricTrainConfig):
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    device = get_device()

    # use same split logic; val acts as test
    (_, _), (test_ds, test_loader), test_ids = create_split_pyg_loaders(
        config, device, train_ratio=0.8
    )

    ckpt_path = const.ARTIFACTS_DIR / f"{config.model_name}_best_weights.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = model.to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    test_loss, test_mse = evaluate(model, test_loader, device)

    print(f"Model: {config.model_name}")
    print(f"Test Loss: {test_loss:.4f} | Test MSE: {test_mse:.6f}")

    results = {
        "model": config.model_name,
        "test_loss": test_loss,
        "test_mse": test_mse,
    }
    out_path = const.ARTIFACTS_DIR / f"{config.model_name}_test_results.csv"
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")
    return results

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

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_graphs = 0
    for data in loader:
        data = data.to(device)
        try:
            out = model(data.x, data.edge_index)
        except TypeError:
            out = model(data.x)
        # loss = F.kl_div(F.log_softmax(out, dim=1), data.y, reduction="batchmean")
        loss = kl_loss_area(out, data.y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
        n_graphs += 1
    return total_loss / max(1, n_graphs)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    n_graphs = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x = data.x
            try:
                out = model(data.x, data.edge_index)
            except TypeError:
                out = model(data.x)
            # loss = F.kl_div(F.log_softmax(out, dim=1), data.y, reduction="batchmean")
            loss = kl_loss_area(out, data.y)
            mse = torch.mean((F.softmax(out, dim=1) - data.y) ** 2).item()
            total_loss += float(loss.item())
            total_mse += mse
            n_graphs += 1
    return total_loss / max(1, n_graphs), total_mse / max(1, n_graphs)

def create_split_pyg_loaders(
    config: GeometricTrainConfig,
    device: str | torch.device,
    train_ratio: float = 0.8,
):
    # Load palette and dataset
    names, class_rgb_values, unknown_index = load_class_palette(const.CLASS_CSV)
    base = DeepGlobeDataset(const.TRAIN_DATA_DIR, class_rgb_values)

    n = len(base)
    idx = np.arange(n)
    rng = np.random.default_rng(config.random_seed)
    rng.shuffle(idx)
    n_train = int(train_ratio * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    from torch.utils.data import Subset
    base_train = Subset(base, train_idx.tolist())
    base_val = Subset(base, val_idx.tolist())

    train_ds = SuperpixelGraphDatasetV2(
        base=base_train,
        class_rgb_values=class_rgb_values,
        k_values=config.k_values,
        normalize_targets=True,
        device=device,
        unknown_index=unknown_index
    )
    val_ds = SuperpixelGraphDatasetV2(
        base=base_val,
        class_rgb_values=class_rgb_values,
        k_values=config.k_values,
        normalize_targets=True,
        device=device,
        unknown_index=unknown_index,
    )

    pin_mem = isinstance(device, torch.device) and device.type == "cuda"
    train_loader = PyGDataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.train_workers,
        persistent_workers=True,
        pin_memory=pin_mem,
        prefetch_factor=2
    )
    val_loader = PyGDataLoader(
        val_ds,
        batch_size=config.batch_size,
        num_workers=config.val_workers,
        persistent_workers=True,
        pin_memory=pin_mem,
        prefetch_factor=2
    )

    try:
        base_paths = getattr(base, "image_paths", None)
        val_ids = [Path(base_paths[i]).stem for i in val_idx.tolist()] if base_paths is not None else []
    except Exception:
        val_ids = []

    return (train_ds, train_loader), (val_ds, val_loader), val_ids

def save_checkpoint_and_sidecar(
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    config: GeometricTrainConfig,
    val_image_ids: list[str] | None = None,
    state_dict_only: bool = False,
):
    """Save checkpoint and a sidecar JSON with architecture and train config."""
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
    }
    torch.save(payload, ckpt_path)

    # Sidecar JSON path, mirrors train.py naming convention
    sidecar = ckpt_path.with_suffix(".json")
    arch = model.get_architecture()
    meta = {
        "architecture": arch,
        "train_config": getattr(config, "__dict__", dict(config.__class__.__dict__)),
        "val_image_ids": (val_image_ids or []),
    }
    with open(sidecar, "w") as f:
        json.dump(meta, f, indent=2)

def train_model(
    model: torch.nn.Module,
    train_loader: PyGDataLoader,
    val_loader: PyGDataLoader,
    device: torch.device,
    config: GeometricTrainConfig,
    val_image_ids: list[str],
):
    """
    Generic training routine for PyG models using KL-div loss on soft targets.
    Saves best and last checkpoints under `artifacts/` using the model class name.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    ckpt_dir = const.ARTIFACTS_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"{config.model_name}_best.ckpt"
    last_ckpt = ckpt_dir / f"{config.model_name}_last.ckpt"
    best_weights = ckpt_dir / f"{config.model_name}_best_weights.pt"

    best_val = float("inf")
    last_epoch = 0
    metrics_list = []
    for epoch in range(1, config.epochs + 1):
        print(f"Starting epoch {epoch}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_mse = evaluate(model, val_loader, device)
        metrics_list.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_mse': val_mse})
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint_and_sidecar(
                ckpt_path=best_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val=best_val,
                config=config,
                val_image_ids=val_image_ids,
            )
            torch.save(model.state_dict(), best_weights)
        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | mse {val_mse:.6f} | best {best_val:.4f}")
        if epoch % 10 == 0:
            pd.DataFrame(metrics_list).to_csv(ckpt_dir / f"{config.model_name}_metrics.csv", index=False)
        last_epoch = epoch

    save_checkpoint_and_sidecar(
        ckpt_path=last_ckpt,
        model=model,
        optimizer=optimizer,
        epoch=last_epoch,
        best_val=best_val,
        config=config,
        val_image_ids=val_image_ids,
    )
    pd.DataFrame(metrics_list).to_csv(ckpt_dir / f"{config.model_name}_metrics.csv", index=False)
    return best_val

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    return torch.device(device)