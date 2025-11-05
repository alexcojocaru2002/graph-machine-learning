import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN2(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=False, normalize=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, add_self_loops=False, normalize=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.drop(x)
        x = self.conv2(x, edge_index)
        return x

    def get_architecture(self) -> dict:
        """Return a JSON-serializable dict describing the model architecture."""
        return {
            "model_class": self.__class__.__name__,
            "in_dim": int(self.in_dim),
            "hidden_dim": int(self.hidden_dim),
            "out_dim": int(self.out_dim),
            "dropout": float(self.dropout),
            "add_self_loops": False,
            "normalize": True,
            "final_activation": "softmax",
        }

    @staticmethod
    def load_model(cfg, num_classes_eff: int) -> "GCN2":
        """
        Build GCN2 based on checkpoint architecture metadata if available,
        otherwise fall back to values in cfg. Also loads checkpoint weights if provided.
        """
        from pathlib import Path
        import json

        device = torch.device(getattr(cfg, "device", "cpu"))
        arch_path = Path(cfg.ckpt_path).with_suffix(".json") if getattr(cfg, "ckpt_path", None) else None
        arch = None
        if arch_path is not None and arch_path.exists():
            with open(arch_path, "r") as f:
                meta = json.load(f)
            arch = meta.get("architecture", None)

        if arch is not None:
            out_dim = arch.get("out_dim", num_classes_eff)
            if out_dim != num_classes_eff:
                out_dim = num_classes_eff
            model = GCN2(
                in_dim=arch.get("in_dim", getattr(cfg, "in_dim", 1024)),
                hidden_dim=arch.get("hidden_dim", getattr(cfg, "hidden_dim", 512)),
                out_dim=out_dim,
                dropout=arch.get("dropout", getattr(cfg, "dropout", 0.2)),
            ).to(device)
        else:
            # Fallback to cfg values if no architecture metadata
            model = GCN2(
                in_dim=getattr(cfg, "in_dim", 1024),
                hidden_dim=getattr(cfg, "hidden_dim", 512),
                out_dim=num_classes_eff,
                dropout=getattr(cfg, "dropout", 0.2),
            ).to(device)

        # Load checkpoint weights if available
        ckpt_path = getattr(cfg, "ckpt_path", None)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=device)
            # Support both full checkpoint dicts and plain state dict files
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                # Assume it's already a state_dict
                state_dict = ckpt
            model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model
