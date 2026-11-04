"""Class-conditional uncertainty-aware prototypes (SADOS, Eqs. 1-5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .partition import normalize_scores


@dataclass
class PrototypeConfig:
    top_m: int = 64
    ema_beta: float = 0.2
    sharpness_gamma: float = 5.0
    reliability_eta: float = 1.0
    reliability_norm: str = "minmax"
    eps: float = 1e-6

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "PrototypeConfig":
        if cfg is None:
            return cls()
        known = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in cfg.items() if key in known})


@dataclass
class PrototypeState:
    """Prototype memory indexed by class and confidence group: [C, K, d]."""

    prototypes: torch.Tensor
    valid_mask: torch.Tensor
    steps: torch.Tensor

    @property
    def num_classes(self) -> int:
        return int(self.prototypes.size(0))

    @property
    def num_groups(self) -> int:
        return int(self.prototypes.size(1))

    @property
    def hidden_dim(self) -> int:
        return int(self.prototypes.size(2))

    def to(self, device: torch.device | str) -> "PrototypeState":
        self.prototypes = self.prototypes.to(device)
        self.valid_mask = self.valid_mask.to(device)
        self.steps = self.steps.to(device)
        return self


@dataclass
class PrototypeUpdateResult:
    state: PrototypeState
    group_reliability_bias: torch.Tensor
    group_mean_confidence: torch.Tensor
    selected_counts: torch.Tensor


def init_prototype_state(
    *,
    num_groups: int,
    hidden_dim: int,
    num_classes: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> PrototypeState:
    shape = (num_classes, num_groups)
    return PrototypeState(
        prototypes=torch.zeros(*shape, hidden_dim, device=device, dtype=dtype),
        valid_mask=torch.zeros(shape, device=device, dtype=torch.bool),
        steps=torch.zeros(shape, device=device, dtype=torch.long),
    )


def update_prototypes(
    embeddings: torch.Tensor,
    unreliable_groups: list[torch.Tensor],
    calibrated_confidence: torch.Tensor,
    pseudo_labels: torch.Tensor,
    classifier_weight: torch.Tensor,
    *,
    state: PrototypeState | None = None,
    config: PrototypeConfig | dict[str, Any] | None = None,
    num_classes: int | None = None,
) -> PrototypeUpdateResult:
    """Select top-M per class/group, then initialize and EMA-update memory."""

    cfg = config if isinstance(config, PrototypeConfig) else PrototypeConfig.from_dict(config)
    if embeddings.dim() != 2:
        raise ValueError("embeddings must have shape [num_nodes, hidden_dim].")
    if cfg.top_m < 1 or not 0 < cfg.ema_beta <= 1:
        raise ValueError("top_m or ema_beta is outside its valid range.")

    device, dtype = embeddings.device, embeddings.dtype
    n_nodes, hidden_dim = embeddings.shape
    pseudo_labels = pseudo_labels.to(device=device, dtype=torch.long).view(-1)
    confidence = calibrated_confidence.to(device=device, dtype=dtype).view(-1)
    classifier_weight = classifier_weight.to(device=device, dtype=dtype)
    classes = int(num_classes if num_classes is not None else classifier_weight.size(0))
    if pseudo_labels.numel() != n_nodes or confidence.numel() != n_nodes:
        raise ValueError("pseudo_labels and calibrated_confidence must cover all nodes.")
    if classifier_weight.shape != (classes, hidden_dim):
        raise ValueError("classifier_weight must have shape [num_classes, hidden_dim].")

    groups = [group.to(device=device, dtype=torch.long).view(-1) for group in unreliable_groups]
    num_groups = len(groups)
    if state is None:
        state = init_prototype_state(
            num_groups=num_groups,
            hidden_dim=hidden_dim,
            num_classes=classes,
            device=device,
            dtype=dtype,
        )
    else:
        state = state.to(device)
        _validate_state(state, classes, num_groups, hidden_dim)

    prototypes = state.prototypes.clone()
    valid = state.valid_mask.clone()
    steps = state.steps.clone()
    selected_counts = torch.zeros(classes, num_groups, device=device, dtype=torch.long)
    beta = float(cfg.ema_beta)

    for group_id, group_idx in enumerate(groups):
        for class_id in range(classes):
            class_idx = group_idx[pseudo_labels[group_idx] == class_id]
            if class_idx.numel() == 0:
                continue
            selected = select_top_confidence(class_idx, confidence, top_m=cfg.top_m)
            selected_counts[class_id, group_id] = selected.numel()
            candidate = aggregate_group_prototype(
                embeddings[selected], confidence[selected], sharpness_gamma=cfg.sharpness_gamma
            )
            if valid[class_id, group_id]:
                previous = prototypes[class_id, group_id]
            else:
                # Initialize the memory from the classifier's class parameter.
                previous = F.normalize(classifier_weight[class_id], dim=0, eps=cfg.eps)
            prototypes[class_id, group_id] = (1.0 - beta) * previous + beta * candidate
            valid[class_id, group_id] = True
            steps[class_id, group_id] += 1

    group_mean = compute_group_mean_confidence(groups, confidence)
    return PrototypeUpdateResult(
        state=PrototypeState(prototypes, valid, steps),
        group_reliability_bias=compute_group_reliability_bias(group_mean, cfg),
        group_mean_confidence=group_mean,
        selected_counts=selected_counts,
    )


def aggregate_group_prototype(
    embeddings: torch.Tensor, confidence: torch.Tensor, *, sharpness_gamma: float
) -> torch.Tensor:
    if embeddings.numel() == 0:
        raise ValueError("Cannot aggregate an empty group.")
    weights = torch.softmax(float(sharpness_gamma) * confidence.view(-1), dim=0)
    return torch.sum(weights.unsqueeze(-1) * embeddings, dim=0)


def compute_group_mean_confidence(
    unreliable_groups: list[torch.Tensor], calibrated_confidence: torch.Tensor
) -> torch.Tensor:
    values = [
        calibrated_confidence[group].mean()
        if group.numel()
        else calibrated_confidence.new_tensor(float("nan"))
        for group in unreliable_groups
    ]
    return torch.stack(values) if values else calibrated_confidence.new_empty(0)


def compute_group_reliability_bias(
    group_mean_confidence: torch.Tensor,
    config: PrototypeConfig | dict[str, Any] | None = None,
) -> torch.Tensor:
    cfg = config if isinstance(config, PrototypeConfig) else PrototypeConfig.from_dict(config)
    normalized = normalize_scores(
        group_mean_confidence, mode=cfg.reliability_norm, eps=cfg.eps
    )
    return float(cfg.reliability_eta) * normalized


def select_top_confidence(
    indices: torch.Tensor, confidence: torch.Tensor, *, top_m: int
) -> torch.Tensor:
    if indices.numel() == 0:
        return indices
    k = min(int(top_m), int(indices.numel()))
    _, order = torch.topk(confidence[indices], k=k, largest=True, sorted=False)
    return indices[order]


def _validate_state(
    state: PrototypeState, num_classes: int, num_groups: int, hidden_dim: int
) -> None:
    shape = (num_classes, num_groups)
    if state.prototypes.shape != (*shape, hidden_dim):
        raise ValueError("PrototypeState has incompatible prototype dimensions.")
    if state.valid_mask.shape != shape or state.steps.shape != shape:
        raise ValueError("PrototypeState has incompatible mask or step dimensions.")
