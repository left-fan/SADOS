"""Confidence calibration and node partitioning for SADOS.

This module implements Appendix C from the paper:

    delta_v = lambda * norm(I(v; N(v)))
    s'_v = sigmoid(logit(s_v) + delta_v)

where the neighborhood agreement score is estimated with an InfoNCE-style
node-neighborhood objective. Calibrated confidence is then used to form the
reliable set and K confidence-stratified unreliable groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
import torch.nn.functional as F


@dataclass
class ConfidenceCalibrationConfig:
    """Controls node-adaptive confidence calibration."""

    lambda_mi: float = 0.2
    temperature: float = 0.2
    normalize: str = "minmax"
    eps: float = 1e-6
    chunk_size: int = 2048
    include_self_in_neighborhood: bool = True
    detach: bool = True

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "ConfidenceCalibrationConfig":
        if cfg is None:
            return cls()
        known = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in cfg.items() if key in known})


@dataclass
class PartitionConfig:
    """Thresholds and calibration settings for SADOS node partitioning."""

    thresholds: tuple[float, ...] = (0.80, 0.60, 0.40, 0.20, 0.00)
    calibration: ConfidenceCalibrationConfig = field(default_factory=ConfidenceCalibrationConfig)
    include_below_last_threshold: bool = True

    @property
    def num_groups(self) -> int:
        return len(self.thresholds) - 1

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "PartitionConfig":
        if cfg is None:
            return cls()
        values = dict(cfg)
        calibration = values.get("calibration")
        if isinstance(calibration, dict):
            values["calibration"] = ConfidenceCalibrationConfig.from_dict(calibration)
        if "thresholds" in values:
            values["thresholds"] = tuple(float(v) for v in values["thresholds"])
        known = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in values.items() if key in known})


@dataclass
class PartitionResult:
    reliable_idx: torch.Tensor
    unreliable_idx: torch.Tensor
    unreliable_groups: list[torch.Tensor]
    pseudo_labels: torch.Tensor
    raw_confidence: torch.Tensor
    calibrated_confidence: torch.Tensor
    mi_score: torch.Tensor
    correction: torch.Tensor
    thresholds: tuple[float, ...]

    def masks(self, num_nodes: int | None = None) -> dict[str, torch.Tensor]:
        if num_nodes is None:
            num_nodes = int(self.pseudo_labels.numel())
        return {
            "reliable_mask": index_to_mask(self.reliable_idx, num_nodes),
            "unreliable_mask": index_to_mask(self.unreliable_idx, num_nodes),
            **{
                f"unreliable_group_{i}_mask": index_to_mask(group, num_nodes)
                for i, group in enumerate(self.unreliable_groups)
            },
        }


def partition_by_confidence(
    *,
    logits: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    pseudo_labels: torch.Tensor | None = None,
    embeddings: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    node_idx: torch.Tensor | None = None,
    config: PartitionConfig | dict[str, Any] | None = None,
    thresholds: Iterable[float] | None = None,
) -> PartitionResult:
    """Calibrate confidence and partition nodes.

    Args:
        logits: Detector logits with shape ``[N, C]`` or binary logits ``[N]``.
        confidence: Optional raw confidence ``s_v``. If omitted, it is derived
            from ``logits`` as the probability of the predicted class.
        pseudo_labels: Optional predicted labels. If omitted, derived from
            ``logits`` or ``confidence``.
        embeddings: Node embeddings ``h_v`` used for MI calibration.
        edge_index: COO graph edges. Required only when MI calibration is used.
        node_idx: Optional global node ids to partition. Confidence calibration
            is still computed on the provided graph, but reliable/unreliable
            sets are restricted to these nodes.
        config: Partition settings.
        thresholds: Optional override for ``config.thresholds``.
    """

    cfg = config if isinstance(config, PartitionConfig) else PartitionConfig.from_dict(config)
    if thresholds is not None:
        cfg.thresholds = tuple(float(v) for v in thresholds)
    _validate_thresholds(cfg.thresholds)

    if confidence is None or pseudo_labels is None:
        if logits is None:
            raise ValueError("Either logits or both confidence and pseudo_labels must be provided.")
        inferred_labels, inferred_conf = confidence_from_logits(logits)
        if confidence is None:
            confidence = inferred_conf
        if pseudo_labels is None:
            pseudo_labels = inferred_labels

    raw_confidence = confidence.float().view(-1)
    pseudo_labels = pseudo_labels.long().view(-1)
    if raw_confidence.numel() != pseudo_labels.numel():
        raise ValueError("confidence and pseudo_labels must have the same number of nodes.")

    if embeddings is not None and edge_index is not None and cfg.calibration.lambda_mi != 0:
        mi_score = neighborhood_agreement_scores(embeddings, edge_index, cfg.calibration)
    else:
        mi_score = torch.zeros_like(raw_confidence)

    calibrated, correction = calibrate_confidence(raw_confidence, mi_score, cfg.calibration)
    reliable_idx, unreliable_idx, groups = stratify_by_thresholds(
        calibrated,
        cfg.thresholds,
        node_idx=node_idx,
        include_below_last_threshold=cfg.include_below_last_threshold,
    )

    if cfg.calibration.detach:
        raw_confidence = raw_confidence.detach()
        pseudo_labels = pseudo_labels.detach()
        mi_score = mi_score.detach()
        calibrated = calibrated.detach()
        correction = correction.detach()

    return PartitionResult(
        reliable_idx=reliable_idx,
        unreliable_idx=unreliable_idx,
        unreliable_groups=groups,
        pseudo_labels=pseudo_labels,
        raw_confidence=raw_confidence,
        calibrated_confidence=calibrated,
        mi_score=mi_score,
        correction=correction,
        thresholds=cfg.thresholds,
    )


def confidence_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predicted labels and confidence from detector logits."""

    if logits.dim() == 1 or logits.size(-1) == 1:
        score = torch.sigmoid(logits.view(-1))
        labels = (score >= 0.5).long()
        confidence = torch.where(labels.bool(), score, 1.0 - score)
        return labels, confidence

    prob = torch.softmax(logits, dim=-1)
    confidence, labels = prob.max(dim=-1)
    return labels.long(), confidence.float()


def calibrate_confidence(
    confidence: torch.Tensor,
    mi_score: torch.Tensor,
    config: ConfidenceCalibrationConfig | dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply logit-space confidence calibration from Eq. 24-26."""

    cfg = (
        config
        if isinstance(config, ConfidenceCalibrationConfig)
        else ConfidenceCalibrationConfig.from_dict(config)
    )
    confidence = confidence.float().view(-1)
    mi_score = mi_score.to(device=confidence.device, dtype=confidence.dtype).view(-1)
    if confidence.numel() != mi_score.numel():
        raise ValueError("confidence and mi_score must have the same length.")

    normalized_mi = normalize_scores(mi_score, mode=cfg.normalize, eps=cfg.eps)
    correction = float(cfg.lambda_mi) * normalized_mi
    clipped = confidence.clamp(cfg.eps, 1.0 - cfg.eps)
    calibrated = torch.sigmoid(torch.logit(clipped) + correction)
    return calibrated, correction


@torch.no_grad()
def neighborhood_agreement_scores(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    config: ConfidenceCalibrationConfig | dict[str, Any] | None = None,
) -> torch.Tensor:
    """Estimate ``I(v; N(v))`` with an InfoNCE-style score.

    The returned value is the positive node-neighborhood similarity logit minus
    the log-sum-exp over a mini-batch neighborhood bank. This bounds memory to
    O(B²), as allowed by the paper's mini-batch InfoNCE approximation.
    """

    cfg = (
        config
        if isinstance(config, ConfidenceCalibrationConfig)
        else ConfidenceCalibrationConfig.from_dict(config)
    )
    if embeddings.dim() != 2:
        raise ValueError("embeddings must have shape [num_nodes, hidden_dim].")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    h = embeddings.detach() if cfg.detach else embeddings
    h = h.float()
    neigh = mean_neighborhood_embeddings(
        h,
        edge_index,
        include_self=cfg.include_self_in_neighborhood,
    )
    h = F.normalize(h, dim=-1, eps=cfg.eps)
    neigh = F.normalize(neigh, dim=-1, eps=cfg.eps)

    num_nodes = h.size(0)
    result = torch.empty(num_nodes, device=h.device, dtype=h.dtype)
    chunk_size = max(1, int(cfg.chunk_size))
    temperature = max(float(cfg.temperature), cfg.eps)

    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        logits = h[start:end] @ neigh[start:end].t()
        logits = logits / temperature
        local = torch.arange(end - start, device=h.device)
        positive = logits.gather(1, local.view(-1, 1)).squeeze(1)
        result[start:end] = positive - torch.logsumexp(logits, dim=1)

    return result


def mean_neighborhood_embeddings(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    include_self: bool = True,
) -> torch.Tensor:
    """Mean-pool source-node embeddings into each target node neighborhood."""

    if embeddings.dim() != 2:
        raise ValueError("embeddings must have shape [num_nodes, hidden_dim].")
    num_nodes = embeddings.size(0)
    edge_index = edge_index.to(device=embeddings.device, dtype=torch.long)
    src, dst = edge_index[0], edge_index[1]

    agg = torch.zeros_like(embeddings)
    deg = torch.zeros(num_nodes, device=embeddings.device, dtype=embeddings.dtype)
    if src.numel() > 0:
        agg.index_add_(0, dst, embeddings[src])
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=embeddings.dtype))

    if include_self:
        agg = agg + embeddings
        deg = deg + 1.0

    isolated = deg <= 0
    deg = deg.clamp_min(1.0).unsqueeze(-1)
    out = agg / deg
    if isolated.any():
        out[isolated] = embeddings[isolated]
    return out


def stratify_by_thresholds(
    calibrated_confidence: torch.Tensor,
    thresholds: Iterable[float],
    *,
    node_idx: torch.Tensor | None = None,
    include_below_last_threshold: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Build reliable set and ordered unreliable confidence groups."""

    tau = tuple(float(v) for v in thresholds)
    _validate_thresholds(tau)
    score = calibrated_confidence.view(-1)
    if node_idx is None:
        candidates = torch.arange(score.numel(), device=score.device, dtype=torch.long)
    else:
        candidates = node_idx.to(device=score.device, dtype=torch.long).view(-1)
    candidate_score = score[candidates]
    reliable = candidates[torch.nonzero(candidate_score >= tau[0], as_tuple=False).view(-1)]
    unreliable = candidates[torch.nonzero(candidate_score < tau[0], as_tuple=False).view(-1)]

    groups: list[torch.Tensor] = []
    for i in range(1, len(tau)):
        upper = tau[i - 1]
        lower = tau[i]
        if i == len(tau) - 1 and include_below_last_threshold:
            mask = candidate_score < upper
        else:
            mask = (candidate_score < upper) & (candidate_score >= lower)
        groups.append(candidates[torch.nonzero(mask, as_tuple=False).view(-1)])

    return reliable.long(), unreliable.long(), groups


def normalize_scores(
    scores: torch.Tensor,
    *,
    mode: str = "minmax",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize a score vector while preserving device and dtype."""

    scores = scores.float()
    finite = torch.isfinite(scores)
    if not finite.any():
        return torch.zeros_like(scores)

    safe = torch.where(finite, scores, torch.zeros_like(scores))
    mode = mode.lower()
    if mode == "none":
        out = safe
    elif mode == "minmax":
        valid = safe[finite]
        lo, hi = valid.min(), valid.max()
        out = (safe - lo) / (hi - lo + eps)
    elif mode == "zscore":
        valid = safe[finite]
        out = (safe - valid.mean()) / (valid.std(unbiased=False) + eps)
    elif mode == "sigmoid_zscore":
        valid = safe[finite]
        out = torch.sigmoid((safe - valid.mean()) / (valid.std(unbiased=False) + eps))
    elif mode == "rank":
        order = torch.argsort(torch.argsort(safe))
        out = order.to(dtype=scores.dtype) / max(1, scores.numel() - 1)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return torch.where(finite, out, torch.zeros_like(out))


def index_to_mask(indices: torch.Tensor, num_nodes: int) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=indices.device)
    if indices.numel() > 0:
        mask[indices.long()] = True
    return mask


def _validate_thresholds(thresholds: tuple[float, ...]) -> None:
    if len(thresholds) < 2:
        raise ValueError("At least two thresholds are required: tau_L0 and tau_L1.")
    for left, right in zip(thresholds, thresholds[1:]):
        if left <= right:
            raise ValueError("thresholds must be strictly decreasing.")
    if thresholds[0] <= 0 or thresholds[0] >= 1:
        raise ValueError("The reliable threshold tau_L0 must be in (0, 1).")
