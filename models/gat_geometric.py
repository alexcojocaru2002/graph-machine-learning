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
