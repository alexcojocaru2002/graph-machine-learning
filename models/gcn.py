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
        }
