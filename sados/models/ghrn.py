"""Graph heterophily reduction from baseline class predictions."""

from __future__ import annotations

import hashlib

import torch

from .bwgnn import BWGNNDetector, normalized_adjacency


class GHRNDetector(BWGNNDetector):
    """BWGNN trained afresh on a graph pruned by post aggregation."""


GHRN = GHRNDetector


@torch.no_grad()
def post_aggregation(
    edge_index: torch.Tensor, probabilities: torch.Tensor,
    *, normalization: str = "symmetric",
) -> torch.Tensor:
    """Subtract neighbor aggregated class probabilities from each node."""
    num_nodes = probabilities.size(0)
    if normalization == "symmetric":
        adjacency = normalized_adjacency(edge_index, num_nodes, probabilities.dtype)
        neighbor = torch.sparse.mm(adjacency, probabilities)
    elif normalization == "left":
        edges = edge_index[:, edge_index[0] != edge_index[1]]
        loops = torch.arange(num_nodes, device=edge_index.device)
        src = torch.cat((edges[0], loops))
        dst = torch.cat((edges[1], loops))
        neighbor = torch.zeros_like(probabilities)
        neighbor.index_add_(0, dst, probabilities[src])
        degree = torch.bincount(dst, minlength=num_nodes).to(probabilities.dtype)
        neighbor = neighbor / degree.clamp_min(1).unsqueeze(1)
    else:
        raise ValueError("normalization must be symmetric or left.")
    return probabilities - neighbor


@torch.no_grad()
def prune_edges_by_pa(
    edge_index: torch.Tensor, probabilities: torch.Tensor, *,
    delete_ratio: float = 0.015, normalization: str = "symmetric",
    preserve_undirected: bool = True,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Drop edges whose endpoints have the lowest PA inner product."""
    if not 0 <= delete_ratio < 1:
        raise ValueError("delete_ratio must lie in [0, 1).")
    if edge_index.size(1) == 0:
        return edge_index.clone(), {"edges_before": 0, "edges_after": 0, "deleted": 0}
    pa = post_aggregation(edge_index, probabilities, normalization=normalization)
    src, dst = edge_index
    scores = (pa[src] * pa[dst]).sum(dim=-1)
    if preserve_undirected:
        n = probabilities.size(0)
        key = torch.minimum(src, dst) * n + torch.maximum(src, dst)
        unique_key, inverse = torch.unique(key, sorted=True, return_inverse=True)
        group_score = torch.zeros(unique_key.numel(), device=scores.device, dtype=scores.dtype)
        group_score.index_add_(0, inverse, scores)
        group_count = torch.bincount(inverse, minlength=unique_key.numel())
        group_score /= group_count.to(scores.dtype)
        count = int(delete_ratio * unique_key.numel())
        remove_group = torch.zeros(unique_key.numel(), dtype=torch.bool, device=scores.device)
        if count:
            remove_group[torch.argsort(group_score, stable=True)[:count]] = True
        keep = ~remove_group[inverse]
    else:
        count = int(delete_ratio * scores.numel())
        keep = torch.ones(scores.numel(), dtype=torch.bool, device=scores.device)
        if count:
            keep[torch.argsort(scores, stable=True)[:count]] = False
    pruned = edge_index[:, keep]
    return pruned, {
        "edges_before": int(edge_index.size(1)),
        "edges_after": int(pruned.size(1)),
        "deleted": int((~keep).sum().item()),
        "delete_ratio": float(delete_ratio),
    }


def graph_signature(graph) -> str:
    """Identify the original graph and split without hashing any labels."""
    digest = hashlib.sha256()
    digest.update(str(tuple(graph.x.shape)).encode())
    for tensor in (
        graph.x, graph.edge_index, graph.train_mask, graph.val_mask, graph.test_mask,
    ):
        if tensor is None:
            digest.update(b"none")
        else:
            arr = tensor.detach().cpu().contiguous().numpy()
            digest.update(str(arr.shape).encode())
            digest.update(arr.tobytes())
    return digest.hexdigest()
