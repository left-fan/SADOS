"""Bernstein wavelet graph neural network in pure PyTorch."""

from __future__ import annotations

from math import comb

import torch
from torch import nn
import torch.nn.functional as F


def bernstein_coefficients(order: int) -> torch.Tensor:
    """Polynomial coefficients for (d+1) C(d,i) (L/2)^i (I-L/2)^(d-i)."""
    if order < 1:
        raise ValueError("Bernstein order must be at least one.")
    coefficients = torch.zeros(order + 1, order + 1, dtype=torch.float64)
    for i in range(order + 1):
        for k in range(i, order + 1):
            coefficients[i, k] = (
                (order + 1) * comb(order, i) * comb(order - i, k - i)
                * (-1) ** (k - i) / (2 ** k)
            )
    return coefficients.float()


def normalized_adjacency(
    edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype,
) -> torch.Tensor:
    """D_out^-1/2 (A+I) D_in^-1/2, with one loop at each node."""
    device = edge_index.device
    edges = edge_index[:, edge_index[0] != edge_index[1]]
    loops = torch.arange(num_nodes, device=device)
    src = torch.cat((edges[0], loops))
    dst = torch.cat((edges[1], loops))
    values = torch.ones(src.numel(), dtype=dtype, device=device)
    out_degree = torch.zeros(num_nodes, dtype=dtype, device=device)
    in_degree = torch.zeros_like(out_degree)
    out_degree.scatter_add_(0, src, values)
    in_degree.scatter_add_(0, dst, values)
    values = values * out_degree[src].rsqrt() * in_degree[dst].rsqrt()
    return torch.sparse_coo_tensor(
        torch.stack((dst, src)), values, (num_nodes, num_nodes),
        device=device, dtype=dtype,
    ).coalesce()


class BWGNNDetector(nn.Module):
    """Original BWGNN filter bank with a binary classification head."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        order = int(num_layers)
        self.order = order
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels)
        self.linear3 = nn.Linear((order + 1) * hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.register_buffer("coefficients", bernstein_coefficients(order))
        self.dropout = float(dropout)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.linear1(x))
        h = F.relu(self.linear2(h))
        adjacency = normalized_adjacency(edge_index, x.size(0), h.dtype)
        powers = [h]
        for _ in range(self.order):
            previous = powers[-1]
            powers.append(previous - torch.sparse.mm(adjacency, previous))
        stacked = torch.stack(powers)
        filters = torch.einsum(
            "ik,knh->nih", self.coefficients.to(dtype=h.dtype), stacked
        )
        h = F.relu(self.linear3(filters.reshape(x.size(0), -1)))
        return F.dropout(h, p=self.dropout, training=self.training)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, *,
        return_embedding: bool = False,
    ):
        embedding = self.encode(x, edge_index)
        logits = self.classifier(embedding)
        return (logits, embedding) if return_embedding else logits

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.forward(x, edge_index).softmax(dim=-1)[:, 1]


BWGNN = BWGNNDetector
