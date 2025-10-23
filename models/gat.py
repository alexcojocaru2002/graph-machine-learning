import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    A PyTorch implementation of a single-head GAT layer without relying on external GNN libraries.

    Inputs:
      - node_features: FloatTensor [N, in_dim]
      - edge_index: LongTensor [2, E] with source->target directed edges

    Returns:
      - out: FloatTensor [N, out_dim]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        bias: bool = True,
        add_self_loops: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(2 * out_dim))  # a in paper for [Wh_i || Wh_j]
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = node_features.size(0)
        assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index must be [2, E]"

        h = self.linear(node_features)  # [N, out_dim]

        if self.add_self_loops:
            self_loops = torch.arange(N, device=h.device)
            edge_index = torch.cat(
                [edge_index, torch.stack((self_loops, self_loops), dim=0)], dim=1
            )

        # Compute unnormalized attention coefficients for each edge via concatenation
        src, dst = edge_index[0], edge_index[1]  # [E]
        h_src = h[src]  # [E, out_dim]
        h_dst = h[dst]  # [E, out_dim]
        e_input = torch.cat([h_src, h_dst], dim=-1)  # [E, 2*out_dim]
        e = (e_input * self.attn).sum(dim=-1)
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        # Normalize with softmax over incoming edges per node (dst as the recipient)
        # We compute softmax in a numerically stable way: subtract per-dst max before exp
        attention = self._softmax_segment(e, dst, N)  # [E]
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        # Aggregate messages
        out = torch.zeros((N, self.out_dim), device=h.device, dtype=h.dtype)
        # Ensure source dtype matches destination for AMP compatibility
        msg = (attention.to(h_src.dtype).unsqueeze(-1) * h_src)
        out.index_add_(0, dst, msg)  # sum_j alpha_ij * h_j

        if self.bias is not None:
            out = out + self.bias
        return out

    @staticmethod
    def _softmax_segment(scores: torch.Tensor, dst_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        # scores: [E], dst_index: [E]
        # Vectorized per-destination softmax with scatter_reduce (PyTorch >= 2.0),
        # fallback to global-shifted softmax if unavailable.
        max_per_dst = torch.full((num_nodes,), -float("inf"), device=scores.device, dtype=scores.dtype)
        try:
            max_per_dst.scatter_reduce_(0, dst_index, scores, reduce='amax', include_self=True)
            shift = max_per_dst[dst_index]
            exp_scores = torch.exp(scores - shift)
            sum_per_dst = torch.zeros((num_nodes,), device=scores.device, dtype=scores.dtype)
            sum_per_dst.index_add_(0, dst_index, exp_scores)
            denom = sum_per_dst[dst_index] + 1e-9
            return exp_scores / denom
        except Exception:
            gmax = torch.max(scores)
            exp_scores = torch.exp(scores - gmax)
            sum_per_dst = torch.zeros((num_nodes,), device=scores.device, dtype=scores.dtype)
            sum_per_dst.index_add_(0, dst_index, exp_scores)
            denom = sum_per_dst[dst_index] + 1e-9
            return exp_scores / denom


class MultiHeadGATLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        concat: bool = True,
        add_self_loops: bool = False,
    ) -> None:
        super().__init__()
        self.concat = concat
        self.heads = nn.ModuleList(
            [GraphAttentionLayer(in_dim, out_dim, dropout, negative_slope, add_self_loops=add_self_loops) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        head_outputs = [head(x, edge_index) for head in self.heads]
        if self.concat:
            return torch.cat(head_outputs, dim=-1)
        else:
            return torch.mean(torch.stack(head_outputs, dim=0), dim=0)


class MultiLayerIntegratedGAT(nn.Module):
    """
    GAT with multi-layer integration as described in MLRSSC-CNN-GNN, adapted for node-level regression.

    We collect representations from each intermediate GAT block and fuse them via concatenation
    followed by a linear projection. A residual connection from the input is added when dimensions match.
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

        self.dropout = dropout
        self.activation = activation

        layers: List[nn.Module] = []
        norms: List[nn.Module] = []
        dims: List[int] = []

        # First layer expands/adjusts dimension
        layers.append(MultiHeadGATLayer(in_dim, hidden_dim, num_heads=num_heads, dropout=dropout, concat=True, add_self_loops=add_self_loops))
        dims.append(hidden_dim * num_heads)
        norms.append(nn.BatchNorm1d(dims[-1]) if use_batchnorm else nn.Identity())

        # Hidden layers
        for _ in range(1, num_layers):
            layers.append(MultiHeadGATLayer(dims[-1], hidden_dim, num_heads=num_heads, dropout=dropout, concat=True, add_self_loops=add_self_loops))
            dims.append(hidden_dim * num_heads)
            norms.append(nn.BatchNorm1d(dims[-1]) if use_batchnorm else nn.Identity())

        self.gat_layers = nn.ModuleList(layers)
        self.norm_layers = nn.ModuleList(norms)

        # Multi-layer integration via SUM across layers as per paper
        fused_dim = dims[-1]
        self.fusion = nn.Sequential(
            nn.Dropout(integrate_dropout),
            nn.Linear(fused_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        representations: List[torch.Tensor] = []
        h = x
        for layer, norm in zip(self.gat_layers, self.norm_layers):
            h = layer(h, edge_index)
            if self.activation == "relu":
                h = F.relu(h)
            elif self.activation == "elu":
                h = F.elu(h)
            h = norm(h)
            representations.append(h)

        # Sum integration
        h_sum = representations[0]
        for r in representations[1:]:
            h_sum = h_sum + r
        out = self.fusion(h_sum)
        return out


class SPNodeRegressor(nn.Module):
    """
    Wrapper model: takes superpixel features [N, F_in], graph edges, predicts per-node values [N, C_out].
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        integrate_dropout: float = 0.2,
        activation: str = "relu",
        final_activation: Optional[str] = "relu",
        add_self_loops: bool = False,
    ) -> None:
        super().__init__()
        self.core = MultiLayerIntegratedGAT(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            integrate_dropout=integrate_dropout,
            activation=activation,
            add_self_loops=add_self_loops,
        )
        self.final_activation = final_activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        y = self.core(x, edge_index)
        if self.final_activation == "relu":
            y = F.relu(y)
        elif self.final_activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.final_activation == "softmax":
            # per-node distribution over classes
            y = F.softmax(y, dim=-1)
        return y


