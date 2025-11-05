import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

import const
from dataset_loader import DeepGlobeDataset
from datasets.superpixel_graph_dataset_v2 import SuperpixelGraphDatasetV2
from load_palette import load_class_palette
from models.gcn import GCN2

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_graphs = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        loss = F.kl_div(F.log_softmax(out, dim=1), data.y, reduction="batchmean")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_graphs += 1
    return total_loss / max(1, n_graphs)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_graphs = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            loss = F.kl_div(F.log_softmax(out, dim=1), data.y, reduction="batchmean")
            total_loss += float(loss.item())
            n_graphs += 1
    return total_loss / max(1, n_graphs)

def create_pyg_loader(data_dir: str, class_rgb_values: list, unknown_index, k_values: list[int], device: str | torch.device):
    """
    Create a PyTorch Geometric loader from a data directory. Returns the dataset object and the loader in a tuple.

    """
    base = DeepGlobeDataset(data_dir, class_rgb_values)
    ds = SuperpixelGraphDatasetV2(
        base=base,
        class_rgb_values=class_rgb_values,
        k_values=k_values,
        normalize_targets=True,
        device=device,
        unknown_index=unknown_index
    )
    loader = PyGDataLoader(ds, batch_size=const.BATCH_SIZE)

    return ds, loader

def train_model(
    model: torch.nn.Module,
    train_loader: PyGDataLoader,
    val_loader: PyGDataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    model_name: str | None = None,
):
    """
    Generic training routine for PyG models using KL-div loss on soft targets.
    Saves best and last checkpoints under `artifacts/` using the model class name.
    """
    model_name = (model_name or model.__class__.__name__).lower()
    ckpt_dir = const.ARTIFACTS_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"{model_name}_best.ckpt"
    last_ckpt = ckpt_dir / f"{model_name}_last.ckpt"
    best_weights = ckpt_dir / f"{model_name}_best_weights.pt"

    best_val = float("inf")
    last_epoch = 0
    for epoch in range(1, epochs + 1):
        print(f"Starting epoch {epoch}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
            }, best_ckpt)
            torch.save(model.state_dict(), best_weights)
        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | best {best_val:.4f}")
        last_epoch = epoch

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": last_epoch,
        "best_val": best_val,
    }, last_ckpt)

    return best_val

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load palette and dataset
    names, class_rgb_values, unknown_index = load_class_palette(const.CLASS_CSV)

    train_ds, train_loader = create_pyg_loader(const.TRAIN_DATA_DIR, class_rgb_values, unknown_index, const.K_VALUES_GCN, device)
    test_ds, test_loader = create_pyg_loader(const.TEST_DATA_DIR, class_rgb_values, unknown_index, const.K_VALUES_GCN, device)

    # Initialize model from a sample to infer dims
    sample: Data = train_ds[0]
    in_dim = sample.x.size(1)
    out_dim = sample.y.size(1)

    model = GCN2(in_dim=in_dim, hidden_dim=const.HIDDEN_DIM, out_dim=out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=const.LR)

    _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        device=device,
        epochs=const.EPOCHS,
    )

    print("Done.")


if __name__ == "__main__":
    main()
