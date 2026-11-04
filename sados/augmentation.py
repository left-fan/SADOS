"""Confidence-aware shifted augmentation for SADOS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import torch
import torch.nn.functional as F

from .prototype import PrototypeState


@dataclass
class ShiftedAugmentationConfig:
    """Hyperparameters for Eq. 6-10."""

    enabled: bool = True
    rho: float = 0.2
    normal_label: int = 0
    augment_only_normal: bool = True
    normalize_attention_inputs: bool = False
    min_valid_prototypes: int = 1
    eps: float = 1e-6

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "ShiftedAugmentationConfig":
        if cfg is None:
            return cls()
        known = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in cfg.items() if key in known})


@dataclass
class ShiftedAugmentationResult:
    indices: torch.Tensor
    embeddings: torch.Tensor
    labels: torch.Tensor
    attention: torch.Tensor
    valid_group_mask: torch.Tensor

    @property
    def is_empty(self) -> bool:
        return int(self.indices.numel()) == 0


def shifted_augmentation(
    embeddings: torch.Tensor,
    reliable_idx: torch.Tensor,
    pseudo_labels: torch.Tensor,
    prototype_state: PrototypeState,
    group_reliability_bias: torch.Tensor | None = None,
    *,
    config: ShiftedAugmentationConfig | dict[str, Any] | None = None,
) -> ShiftedAugmentationResult:
    """Generate shifted reliable-node embeddings with prototype attention."""

    cfg = (
        config
        if isinstance(config, ShiftedAugmentationConfig)
        else ShiftedAugmentationConfig.from_dict(config)
    )
    if embeddings.dim() != 2:
        raise ValueError("embeddings must have shape [num_nodes, hidden_dim].")

    device = embeddings.device
    dtype = embeddings.dtype
    state = prototype_state.to(device)
    reliable_idx = reliable_idx.to(device=device, dtype=torch.long).view(-1)
    pseudo_labels = pseudo_labels.to(device=device, dtype=torch.long).view(-1)

    if group_reliability_bias is None:
        group_reliability_bias = torch.zeros(state.num_groups, device=device, dtype=dtype)
    else:
        group_reliability_bias = group_reliability_bias.to(device=device, dtype=dtype).view(-1)
        if group_reliability_bias.numel() != state.num_groups:
            raise ValueError("group_reliability_bias must have one value per prototype group.")

    if reliable_idx.numel() == 0:
        return _empty_result(device, dtype, state.num_groups, embeddings.size(1))

    labels = pseudo_labels[reliable_idx]
    if cfg.augment_only_normal:
        keep = labels == int(cfg.normal_label)
        reliable_idx = reliable_idx[keep]
        labels = labels[keep]

    if reliable_idx.numel() == 0:
        return _empty_result(device, dtype, state.num_groups, embeddings.size(1))

    valid_groups = state.valid_mask[labels]
    enough = valid_groups.sum(dim=1) >= cfg.min_valid_prototypes
    reliable_idx, labels, valid_groups = reliable_idx[enough], labels[enough], valid_groups[enough]
    if reliable_idx.numel() == 0:
        return _empty_result(device, dtype, state.num_groups, embeddings.size(1))

    prototypes = state.prototypes[labels]
    queries = embeddings[reliable_idx]

    if cfg.normalize_attention_inputs:
        query_for_score = F.normalize(queries, dim=-1, eps=cfg.eps)
        proto_for_score = F.normalize(prototypes, dim=-1, eps=cfg.eps)
    else:
        query_for_score = queries
        proto_for_score = prototypes

    content = torch.einsum("rd,rkd->rk", query_for_score, proto_for_score)
    content = content / math.sqrt(max(1, embeddings.size(1)))
    logits = (content + group_reliability_bias.view(1, -1)).masked_fill(
        ~valid_groups, -torch.inf
    )
    local_attention = torch.softmax(logits, dim=-1)
    offset = torch.einsum("rk,rkd->rd", local_attention, prototypes)
    augmented = queries + float(cfg.rho) * offset

    return ShiftedAugmentationResult(
        indices=reliable_idx,
        embeddings=augmented,
        labels=labels,
        attention=local_attention,
        valid_group_mask=valid_groups,
    )


def build_supervised_tuning_set(
    embeddings: torch.Tensor,
    reliable_idx: torch.Tensor,
    pseudo_labels: torch.Tensor,
    augmentation: ShiftedAugmentationResult | None = None,
    *,
    normal_only: bool = True,
    normal_label: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use every reliable original plus shifted reliable-normal samples (Eq. 14).

    Args:
        normal_only: When true, restrict the supervised set to normal-labelled
            reliable nodes and their (normal) augmented counterparts.
        normal_label: Label id treated as "normal" for the filter.
    """

    reliable_idx = reliable_idx.to(device=embeddings.device, dtype=torch.long).view(-1)
    if normal_only:
        reliable_labels = pseudo_labels[reliable_idx].long()
        reliable_idx = reliable_idx[reliable_labels == int(normal_label)]
    original = embeddings[reliable_idx]
    original_labels = pseudo_labels[reliable_idx].long()
    if augmentation is None or augmentation.is_empty:
        return original, original_labels, torch.zeros(
            original_labels.numel(), device=embeddings.device, dtype=torch.bool
        )
    augmented_embeddings = augmentation.embeddings
    augmented_labels = augmentation.labels
    if normal_only:
        keep = augmented_labels.long() == int(normal_label)
        augmented_embeddings = augmented_embeddings[keep]
        augmented_labels = augmented_labels[keep]
    features = torch.cat([original, augmented_embeddings], dim=0)
    labels = torch.cat([original_labels, augmented_labels], dim=0)
    is_augmented = torch.cat([
        torch.zeros(original_labels.numel(), device=embeddings.device, dtype=torch.bool),
        torch.ones(augmented_labels.numel(), device=embeddings.device, dtype=torch.bool),
    ])
    return features, labels, is_augmented


def confidence_aware_shifted_augmentation(*args, **kwargs) -> ShiftedAugmentationResult:
    return shifted_augmentation(*args, **kwargs)


def _empty_result(
    device: torch.device,
    dtype: torch.dtype,
    num_groups: int,
    hidden_dim: int,
) -> ShiftedAugmentationResult:
    return ShiftedAugmentationResult(
        indices=torch.empty(0, device=device, dtype=torch.long),
        embeddings=torch.empty(0, hidden_dim, device=device, dtype=dtype),
        labels=torch.empty(0, device=device, dtype=torch.long),
        attention=torch.empty(0, num_groups, device=device, dtype=dtype),
        valid_group_mask=torch.empty(0, num_groups, device=device, dtype=torch.bool),
    )
