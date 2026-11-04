"""Pure PyTorch GCN detector used for source pretraining."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return torch.sparse.mm(adj_norm, x)


class GCNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        dims = [in_channels] + [hidden_channels] * num_layers
        self.layers = nn.ModuleList(
            GCNLayer(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        adj_norm = normalized_adjacency(edge_index, x.size(0), x.device, x.dtype)
        h = x
        for layer_id, layer in enumerate(self.layers):
            h = layer(h, adj_norm)
            if layer_id != len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GCNDetector(nn.Module):
    """GCN encoder plus a binary anomaly classifier."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, num_layers, dropout)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        return_embedding: bool = False,
    ):
        h = self.encode(x, edge_index)
        logits = self.classifier(h)
        if return_embedding:
            return logits, h
        return logits

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, edge_index)
        prob = torch.softmax(logits, dim=-1)
        if prob.size(-1) == 1:
            return prob.squeeze(-1)
        return prob[:, 1]


GCN = GCNDetector


def normalized_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build D^-1/2 (A + I) D^-1/2 as a sparse COO tensor."""

    edge_index = edge_index.to(device=device, dtype=torch.long)
    self_loops = torch.arange(num_nodes, device=device, dtype=torch.long).unsqueeze(0)
    self_loops = self_loops.repeat(2, 1)
    edge_index = torch.cat([edge_index, self_loops], dim=1)

    src, dst = edge_index[0], edge_index[1]
    row, col = dst, src
    values = torch.ones(row.numel(), device=device, dtype=dtype)
    degree = torch.zeros(num_nodes, device=device, dtype=dtype)
    degree.scatter_add_(0, row, values)
    degree_inv_sqrt = degree.clamp_min(1.0).pow(-0.5)
    values = degree_inv_sqrt[row] * values * degree_inv_sqrt[col]

    adj = torch.sparse_coo_tensor(
        torch.stack([row, col], dim=0),
        values,
        (num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    )
    return adj.coalesce()
