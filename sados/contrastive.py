"""Semantic/structural contrastive learning for SADOS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ContrastiveConfig:
    """Hyperparameters for Eq. 11-13."""

    enabled: bool = True
    feature_noise_epsilon: float = 0.1
    edge_drop_prob: float = 0.2
    temperature: float = 0.2
    symmetric: bool = False
    normalize: bool = True
    feature_scale: str = "ones"
    max_nodes: int | None = 2048
    negative_sample_size: int | None = 512
    chunk_size: int = 1024
    eps: float = 1e-8

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "ContrastiveConfig":
        if cfg is None:
            return cls()
        known = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in cfg.items() if key in known})


def contrastive_loss(
    encoder: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    unreliable_idx: torch.Tensor,
    *,
    config: ContrastiveConfig | dict[str, Any] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Build semantic/structural views and compute unreliable-node InfoNCE."""

    cfg = config if isinstance(config, ContrastiveConfig) else ContrastiveConfig.from_dict(config)
    unreliable_idx = unreliable_idx.to(device=x.device, dtype=torch.long).view(-1)
    if unreliable_idx.numel() == 0:
        return differentiable_zero(encoder, x)

    if cfg.max_nodes is not None and unreliable_idx.numel() > cfg.max_nodes:
        perm = torch.randperm(unreliable_idx.numel(), device=x.device, generator=generator)
        unreliable_idx = unreliable_idx[perm[: cfg.max_nodes]]

    semantic_x = construct_semantic_feature_view(x, cfg, generator=generator)
    structural_edge_index = construct_structural_edge_view(edge_index, cfg, generator=generator)

    semantic_embeddings = encode_graph(encoder, semantic_x, edge_index)
    structural_embeddings = encode_graph(encoder, x, structural_edge_index)

    return info_nce_loss(
        semantic_embeddings[unreliable_idx],
        structural_embeddings[unreliable_idx],
        config=cfg,
        generator=generator,
    )


def construct_semantic_feature_view(
    x: torch.Tensor,
    config: ContrastiveConfig | dict[str, Any] | None = None,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Feature-noise semantic view from Eq. 11."""

    cfg = config if isinstance(config, ContrastiveConfig) else ContrastiveConfig.from_dict(config)
    scale = feature_noise_scale(x, mode=cfg.feature_scale, eps=cfg.eps)
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return x + float(cfg.feature_noise_epsilon) * noise * scale.view(1, -1)


def construct_structural_edge_view(
    edge_index: torch.Tensor,
    config: ContrastiveConfig | dict[str, Any] | None = None,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random edge-dropping structural view from Eq. 12."""

    cfg = config if isinstance(config, ContrastiveConfig) else ContrastiveConfig.from_dict(config)
    drop_prob = float(cfg.edge_drop_prob)
    if drop_prob <= 0 or edge_index.numel() == 0:
        return edge_index
    if drop_prob >= 1:
        return edge_index[:, :0]
    keep = torch.rand(
        edge_index.size(1),
        device=edge_index.device,
        generator=generator,
    ) >= drop_prob
    return edge_index[:, keep]


def info_nce_loss(
    z_query: torch.Tensor,
    z_key: torch.Tensor,
    *,
    config: ContrastiveConfig | dict[str, Any] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Compute node-level InfoNCE from Eq. 13."""

    cfg = config if isinstance(config, ContrastiveConfig) else ContrastiveConfig.from_dict(config)
    if z_query.shape != z_key.shape:
        raise ValueError("z_query and z_key must have the same shape.")
    if z_query.dim() != 2:
        raise ValueError("z_query and z_key must have shape [num_nodes, hidden_dim].")
    if z_query.size(0) <= 1:
        return z_query.sum() * 0.0 + z_key.sum() * 0.0

    if cfg.normalize:
        z_query = F.normalize(z_query, dim=-1, eps=cfg.eps)
        z_key = F.normalize(z_key, dim=-1, eps=cfg.eps)

    loss = _one_direction_info_nce(z_query, z_key, cfg, generator=generator)
    if cfg.symmetric:
        loss = 0.5 * (loss + _one_direction_info_nce(z_key, z_query, cfg, generator=generator))
    return loss


def encode_graph(
    encoder: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Call a detector, encoder module, or plain callable consistently."""

    if hasattr(encoder, "encode"):
        return encoder.encode(x, edge_index)  # type: ignore[union-attr]
    output = encoder(x, edge_index)
    if isinstance(output, tuple):
        return output[-1]
    return output


def differentiable_zero(
    encoder: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    if isinstance(encoder, nn.Module):
        for parameter in encoder.parameters():
            return parameter.sum() * 0.0
    return x.sum() * 0.0


def feature_noise_scale(x: torch.Tensor, *, mode: str, eps: float) -> torch.Tensor:
    mode = mode.lower()
    if mode == "std":
        return x.std(dim=0, unbiased=False).clamp_min(eps)
    if mode == "mad":
        median = x.median(dim=0).values
        return (x - median).abs().median(dim=0).values.clamp_min(eps)
    if mode == "ones":
        return torch.ones(x.size(1), device=x.device, dtype=x.dtype)
    raise ValueError("feature_scale must be one of: std, mad, ones.")


def _one_direction_info_nce(
    z_query: torch.Tensor,
    z_key: torch.Tensor,
    cfg: ContrastiveConfig,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if cfg.negative_sample_size is not None:
        return _sampled_info_nce(z_query, z_key, cfg, generator=generator)
    return _full_info_nce(z_query, z_key, cfg)


def _full_info_nce(
    z_query: torch.Tensor,
    z_key: torch.Tensor,
    cfg: ContrastiveConfig,
) -> torch.Tensor:
    num_nodes = z_query.size(0)
    chunk_size = max(1, int(cfg.chunk_size))
    labels_all = torch.arange(num_nodes, device=z_query.device)
    total = z_query.new_tensor(0.0)

    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        logits = (z_query[start:end] @ z_key.t()) / max(cfg.temperature, cfg.eps)
        labels = labels_all[start:end]
        total = total + F.cross_entropy(logits, labels, reduction="sum")
    return total / num_nodes


def _sampled_info_nce(
    z_query: torch.Tensor,
    z_key: torch.Tensor,
    cfg: ContrastiveConfig,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    num_nodes = z_query.size(0)
    neg_k = max(1, min(int(cfg.negative_sample_size), num_nodes - 1))
    chunk_size = max(1, int(cfg.chunk_size))
    total = z_query.new_tensor(0.0)

    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        batch_size = end - start
        positive = torch.arange(start, end, device=z_query.device).view(-1, 1)
        negatives = torch.randint(
            0,
            num_nodes,
            (batch_size, neg_k),
            device=z_query.device,
            generator=generator,
        )
        negatives = torch.where(negatives == positive, (negatives + 1) % num_nodes, negatives)
        candidate_ids = torch.cat([positive, negatives], dim=1)
        candidates = z_key[candidate_ids]
        query = z_query[start:end].unsqueeze(1)
        logits = torch.sum(query * candidates, dim=-1) / max(cfg.temperature, cfg.eps)
        labels = torch.zeros(batch_size, device=z_query.device, dtype=torch.long)
        total = total + F.cross_entropy(logits, labels, reduction="sum")

    return total / num_nodes
