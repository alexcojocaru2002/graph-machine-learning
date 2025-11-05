import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class MultiheadGAT(nn.Module):
    """
    PyTorch Geometric implementation of a multi-layer GAT with layer integration via SUM,

    - Each block is a GATConv with 'heads=num_heads' and 'concat=True'.
    - After each block: activation -> BatchNorm1d.
    - Representations across layers are SUMed, followed by a Linear projection (fusion).
    - Outputs probabilities via softmax activation (for area fraction prediction).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        num_heads: int = 3,
        dropout: float = 0.2,
        integrate_dropout: float = 0.2,
        activation: str = "relu",
        add_self_loops: bool = False,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        assert num_layers >= 1

        # Persist constructor args for reproducible architecture export
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.integrate_dropout = integrate_dropout
        self.activation = activation
        self.add_self_loops = add_self_loops
        self.use_batchnorm = use_batchnorm

        layers = []
        norms = []
        dims = []

        # First layer
        layers.append(
            GATConv(
                in_channels=in_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                add_self_loops=add_self_loops,
                concat=True,
            )
        )
        dims.append(hidden_dim * num_heads)
        norms.append(nn.BatchNorm1d(dims[-1]) if use_batchnorm else nn.Identity())

        # Hidden layers
        for _ in range(1, num_layers):
            layers.append(
                GATConv(
                    in_channels=dims[-1],
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                    concat=True,
                )
            )
            dims.append(hidden_dim * num_heads)
            norms.append(nn.BatchNorm1d(dims[-1]) if use_batchnorm else nn.Identity())

        self.gat_layers = nn.ModuleList(layers)
        self.norm_layers = nn.ModuleList(norms)

        # Integration and fusion (SUM -> Linear)
        fused_dim = dims[-1]
        self.fusion = nn.Sequential(
            nn.Dropout(integrate_dropout),
            nn.Linear(fused_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        representations = []
        h = x
        for layer, norm in zip(self.gat_layers, self.norm_layers):
            h = layer(h, edge_index)
            if self.activation == "relu":
                h = F.relu(h)
            elif self.activation == "elu":
                h = F.elu(h)
            h = norm(h)
            representations.append(h)

        h_sum = representations[0]
        for r in representations[1:]:
            h_sum = h_sum + r
        out = self.fusion(h_sum)
        return out

    def get_architecture(self) -> dict:
        """Return a JSON-serializable dict describing the model architecture."""
        return {
            "model_class": self.__class__.__name__,
            "in_dim": int(self.in_dim),
            "hidden_dim": int(self.hidden_dim),
            "out_dim": int(self.out_dim),
            "num_layers": int(self.num_layers),
            "num_heads": int(self.num_heads),
            "dropout": float(self.dropout),
            "integrate_dropout": float(self.integrate_dropout),
            "activation": self.activation,
            "add_self_loops": bool(self.add_self_loops),
            "use_batchnorm": bool(self.use_batchnorm),
            "final_activation": None,
        }

    @staticmethod
    def load_model(cfg, num_classes_eff: int) -> "MultiheadGAT":
        """
        Build MultiheadGAT based on checkpoint  architecture metadata
        """
        from pathlib import Path  # local import to avoid global dependency
        import json  # local import to avoid global dependency

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
            model = MultiheadGAT(
                in_dim=arch.get("in_dim", getattr(cfg, "in_dim", 1024)),
                hidden_dim=arch.get("hidden_dim", getattr(cfg, "hidden_dim", 512)),
                out_dim=out_dim,
                num_layers=arch.get("num_layers", getattr(cfg, "num_layers", 2)),
                num_heads=arch.get("num_heads", getattr(cfg, "num_heads", 3)),
                dropout=arch.get("dropout", getattr(cfg, "gat_dropout", 0.2)),
                integrate_dropout=arch.get("integrate_dropout", getattr(cfg, "integrate_dropout", 0.2)),
                activation=arch.get("activation", "relu"),
                add_self_loops=arch.get("add_self_loops", False),
                use_batchnorm=arch.get("use_batchnorm", True),
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
