"""SADOS test-time adaptation trainer with dynamic objective scheduling."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
import copy
import csv
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .augmentation import (
    ShiftedAugmentationConfig,
    build_supervised_tuning_set,
    shifted_augmentation,
)
from .contrastive import ContrastiveConfig, contrastive_loss
from .data import GraphData
from .metrics import binary_classification_metrics
from .partition import PartitionConfig, partition_by_confidence
from .prototype import PrototypeConfig, PrototypeState, update_prototypes
from .scheduler import dynamic_supervised_weight


@dataclass
class TTAConfig:
    """Configuration for the full SADOS adaptation loop."""

    epochs: int = 100
    mode: str = "single"
    stream_batch_size: int = 256
    stream_shuffle: bool = False
    updates_per_batch: int = 1
    lr: float = 0.001
    weight_decay: float = 0.0
    initial_supervised_weight: float = 0.4
    dynamic_schedule: bool = True
    fixed_supervised_weight: float = 0.5
    fixed_contrastive_weight: float = 0.5
    supervised_enabled: bool = True
    supervised_normal_only: bool = True
    supervised_normal_only_epochs: int = 100
    supervised_anomaly_tail_epochs: int = 0
    adapt_on: str = "test"
    class_weight: bool = False
    threshold: float = 0.5
    clip_grad_norm: float | None = None
    log_every: int = 1
    evaluate_every: int = 1
    monitor: str = ""
    report_metrics_after_fit: bool = True
    checkpoint_path: str | None = None
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    prototype: PrototypeConfig = field(default_factory=PrototypeConfig)
    augmentation: ShiftedAugmentationConfig = field(default_factory=ShiftedAugmentationConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "TTAConfig":
        if cfg is None:
            return cls()
        values = dict(cfg)
        if isinstance(values.get("partition"), dict):
            values["partition"] = PartitionConfig.from_dict(values["partition"])
        if isinstance(values.get("prototype"), dict):
            values["prototype"] = PrototypeConfig.from_dict(values["prototype"])
        if isinstance(values.get("augmentation"), dict):
            values["augmentation"] = ShiftedAugmentationConfig.from_dict(values["augmentation"])
        if isinstance(values.get("contrastive"), dict):
            values["contrastive"] = ContrastiveConfig.from_dict(values["contrastive"])
        known = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in values.items() if key in known})


class SadosTTATrainer:
    """Full SADOS TTA loop.

    Training uses only pseudo-labels and unlabeled target nodes. Ground-truth
    labels, when present, are read only in ``evaluate`` for reporting.
    """

    def __init__(
        self,
        model: nn.Module,
        graph: GraphData,
        config: TTAConfig | dict[str, Any] | None = None,
        *,
        device: torch.device | str = "cpu",
        target_idx: torch.Tensor | np.ndarray | list[int] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.graph = graph.to(device)
        self.cfg = config if isinstance(config, TTAConfig) else TTAConfig.from_dict(config)
        if self.cfg.monitor.lower().startswith("target_"):
            raise ValueError("Target labels cannot be used for TTA checkpoint selection.")
        self.device = torch.device(device)
        self.target_idx = self._resolve_target_idx(target_idx)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.prototype_state: PrototypeState | None = None
        self.best_state: dict[str, torch.Tensor] | None = None

        if not hasattr(self.model, "classifier"):
            raise ValueError("SADOS supervised TTA expects the detector to expose a classifier head.")
        if not hasattr(self.model, "encode"):
            raise ValueError("SADOS contrastive TTA expects the detector to expose encode(x, edge_index).")

    def fit(self) -> dict[str, Any]:
        history: list[dict[str, float]] = []
        best_score = float("nan")
        best_epoch = -1
        best_metrics: dict[str, Any] = {}
        self.best_state = None

        mode = self.cfg.mode.lower()
        if mode in {"stream", "streaming", "online"}:
            steps = self._stream_steps()
        elif mode in {"single", "epoch", "full"}:
            steps = self._single_steps()
        else:
            raise ValueError("tta.mode must be one of: single, stream.")

        total_steps = max(1, len(steps))
        for step_id, step in enumerate(steps, start=1):
            metrics = self.adapt_step(
                target_idx=step["target_idx"],
                step=step_id,
                total_steps=total_steps,
            )
            metrics.update(step["meta"])
            if self.cfg.evaluate_every and step_id % self.cfg.evaluate_every == 0:
                metrics.update(self.evaluate_all())
            history.append(metrics)

            if self.cfg.monitor:
                score = _monitor_score(metrics, self.cfg.monitor)
                if not np.isfinite(best_score) or score > best_score:
                    best_score = score
                    best_epoch = step_id
                    best_metrics = metrics
                    self.best_state = copy.deepcopy(self.model.state_dict())
                    if self.cfg.checkpoint_path and np.isfinite(score):
                        self.save_checkpoint(self.cfg.checkpoint_path, step_id, metrics)

            if self.cfg.log_every and step_id % self.cfg.log_every == 0:
                print(_format_tta_metrics(metrics), flush=True)

        if self.best_state is not None and np.isfinite(best_score):
            self.model.load_state_dict(self.best_state)

        final_metrics = history[-1] if history else {}
        if self.cfg.report_metrics_after_fit:
            final_metrics.update(self.evaluate_all())

        if self.cfg.checkpoint_path:
            self.save_checkpoint(self.cfg.checkpoint_path, total_steps, final_metrics)

        return {
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_metrics": best_metrics,
            "history": history,
        }

    def train_epoch(self, epoch: int) -> dict[str, float]:
        return self.adapt_step(
            target_idx=self.target_idx,
            step=epoch,
            total_steps=max(1, self.cfg.epochs),
        )

    def adapt_step(
        self,
        *,
        target_idx: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        target_idx = target_idx.to(device=self.device, dtype=torch.long).view(-1)

        logits, embeddings = self.model(
            self.graph.x,
            self.graph.edge_index,
            return_embedding=True,
        )
        partition = partition_by_confidence(
            logits=logits.detach(),
            embeddings=embeddings.detach(),
            edge_index=self.graph.edge_index,
            node_idx=target_idx,
            config=self.cfg.partition,
        )
        classifier_weight = getattr(self.model, "classifier").weight.detach()
        augmentation = None
        if self.cfg.augmentation.enabled:
            prototype_result = update_prototypes(
                embeddings.detach(),
                partition.unreliable_groups,
                partition.calibrated_confidence,
                partition.pseudo_labels,
                classifier_weight,
                state=self.prototype_state,
                config=self.cfg.prototype,
                num_classes=self._num_classes(),
            )
            self.prototype_state = prototype_result.state
            augmentation = shifted_augmentation(
                embeddings,
                partition.reliable_idx,
                partition.pseudo_labels,
                self.prototype_state,
                prototype_result.group_reliability_bias,
                config=self.cfg.augmentation,
            )
        # Without shifted augmentation the supervised set collapses to the
        # reliable (high-quality) nodes only, still trained with cross-entropy.
        # Supervised set schedule: normal-class reliable nodes (and their
        # augmentations) are used by default; the final ``anomaly_tail_epochs``
        # steps additionally include anomaly-class reliable nodes.
        normal_only = _use_normal_only_supervision(step, total_steps, self.cfg)
        supervised_embeddings, supervised_labels, is_augmented = build_supervised_tuning_set(
            embeddings,
            partition.reliable_idx,
            partition.pseudo_labels,
            augmentation,
            normal_only=normal_only,
            normal_label=int(self.cfg.augmentation.normal_label),
        )
        if self.cfg.supervised_enabled:
            supervised_loss = self._supervised_loss(supervised_embeddings, supervised_labels)
        else:
            supervised_loss = _model_zero(self.model, self.device)

        if self.cfg.dynamic_schedule:
            supervised_weight = dynamic_supervised_weight(
                step - 1,
                total_steps,
                self.cfg.initial_supervised_weight,
            )
            contrastive_weight = 1.0 - supervised_weight
        else:
            # Schedule ablation: hold both objective weights constant.
            supervised_weight = float(self.cfg.fixed_supervised_weight)
            contrastive_weight = float(self.cfg.fixed_contrastive_weight)

        # Supervised-loss ablation: keep contrastive learning only at unit weight.
        if not self.cfg.supervised_enabled:
            supervised_weight = 0.0
            contrastive_weight = 1.0

        # Contrastive learning can be ablated: when disabled the objective drops
        # the Eq. 11-13 term entirely.
        contrast_loss = None
        if self.cfg.contrastive.enabled:
            contrast_loss = contrastive_loss(
                self.model,
                self.graph.x,
                self.graph.edge_index,
                partition.unreliable_idx,
                config=self.cfg.contrastive,
            )

        loss = supervised_weight * supervised_loss
        if contrast_loss is not None:
            loss = loss + contrastive_weight * contrast_loss
        else:
            contrastive_weight = 0.0
        loss.backward()

        if self.cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
        self.optimizer.step()

        group_counts = [float(group.numel()) for group in partition.unreliable_groups]
        metrics: dict[str, float] = {
            "epoch": float(step),
            "step": float(step),
            "total_steps": float(total_steps),
            "stream_count": float(target_idx.numel()),
            "loss": float(loss.detach().cpu().item()),
            "supervised_loss": float(supervised_loss.detach().cpu().item()),
            "contrastive_loss": (
                float(contrast_loss.detach().cpu().item()) if contrast_loss is not None else 0.0
            ),
            "supervised_weight": float(supervised_weight),
            "contrastive_weight": float(contrastive_weight),
            "reliable_count": float(partition.reliable_idx.numel()),
            "unreliable_count": float(partition.unreliable_idx.numel()),
            "augmented_count": float(augmentation.indices.numel()) if augmentation is not None else 0.0,
            "supervised_count": float(supervised_labels.numel()),
            "supervised_normal_only": 1.0 if normal_only else 0.0,
            "supervised_include_anomaly": 0.0 if normal_only else 1.0,
            "augmented_supervised_count": float(is_augmented.sum().item()),
            "original_supervised_count": float(supervised_labels.numel() - is_augmented.sum().item()),
            "prototype_valid_count": (
                float(self.prototype_state.valid_mask.sum().item())
                if self.prototype_state is not None
                else 0.0
            ),
            "mean_raw_confidence": _safe_mean(partition.raw_confidence[target_idx]),
            "mean_calibrated_confidence": _safe_mean(partition.calibrated_confidence[target_idx]),
        }
        for group_id, count in enumerate(group_counts, start=1):
            metrics[f"unreliable_group_{group_id}_count"] = count
        return metrics

    def _single_steps(self) -> list[dict[str, Any]]:
        epochs = max(1, int(self.cfg.epochs))
        return [
            {
                "target_idx": self.target_idx,
                "meta": {
                    "mode": "single",
                    "single_epoch": float(epoch),
                    "stream_batch": float("nan"),
                    "stream_update": float("nan"),
                },
            }
            for epoch in range(1, epochs + 1)
        ]

    def _stream_steps(self) -> list[dict[str, Any]]:
        batches = self._stream_batches()
        updates_per_batch = max(1, int(self.cfg.updates_per_batch))
        steps: list[dict[str, Any]] = []
        for batch_id, batch_idx in enumerate(batches, start=1):
            for update_id in range(1, updates_per_batch + 1):
                steps.append(
                    {
                        "target_idx": batch_idx,
                        "meta": {
                            "mode": "stream",
                            "single_epoch": float("nan"),
                            "stream_batch": float(batch_id),
                            "stream_update": float(update_id),
                        },
                    }
                )
        return steps

    def _stream_batches(self) -> list[torch.Tensor]:
        idx = self.target_idx
        if idx.numel() == 0:
            return [idx]

        if self.cfg.stream_shuffle:
            order = torch.randperm(idx.numel(), device=idx.device)
            idx = idx[order]

        batch_size = max(1, int(self.cfg.stream_batch_size))
        return [
            idx[start : start + batch_size]
            for start in range(0, int(idx.numel()), batch_size)
        ]

    @torch.no_grad()
    def evaluate_all(self) -> dict[str, float]:
        metrics = _prefix_metrics("target", self.evaluate(self.target_idx))
        if self.graph.train_mask is not None:
            metrics.update(_prefix_metrics("source_train", self.evaluate_mask(self.graph.train_mask)))
        if self.graph.val_mask is not None:
            metrics.update(_prefix_metrics("source_val", self.evaluate_mask(self.graph.val_mask)))
        return metrics

    @torch.no_grad()
    def evaluate_mask(self, mask: torch.Tensor | None) -> dict[str, float]:
        if mask is None or int(mask.sum().item()) == 0:
            return {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan")}
        idx = torch.nonzero(mask, as_tuple=False).view(-1)
        return self.evaluate(idx)

    @torch.no_grad()
    def evaluate(self, idx: torch.Tensor | np.ndarray | list[int]) -> dict[str, float]:
        idx = torch.as_tensor(idx, device=self.device, dtype=torch.long).view(-1)
        if idx.numel() == 0:
            return {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan")}
        self.model.eval()
        logits = self.model(self.graph.x, self.graph.edge_index)
        prob = torch.softmax(logits, dim=-1)
        score = prob[:, 1] if prob.size(-1) > 1 else prob.squeeze(-1)
        y_true = self.graph.y[idx].detach().cpu().numpy()
        y_score = score[idx].detach().cpu().numpy()
        return binary_classification_metrics(
            y_true,
            y_score,
            threshold=self.cfg.threshold,
        )

    @torch.no_grad()
    def anomaly_scores(self) -> torch.Tensor:
        self.model.eval()
        logits = self.model(self.graph.x, self.graph.edge_index)
        prob = torch.softmax(logits, dim=-1)
        return prob[:, 1] if prob.size(-1) > 1 else prob.squeeze(-1)

    def save_checkpoint(self, path: str | Path, epoch: int, metrics: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": _config_to_dict(self.cfg),
            "target_idx": self.target_idx.detach().cpu(),
        }
        if self.prototype_state is not None:
            payload["prototype_state"] = {
                "prototypes": self.prototype_state.prototypes.detach().cpu(),
                "valid_mask": self.prototype_state.valid_mask.detach().cpu(),
                "steps": self.prototype_state.steps.detach().cpu(),
            }
        torch.save(payload, path)

    def save_history_csv(self, path: str | Path, history: list[dict[str, float]]) -> None:
        if not history:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in history for key in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

    def _supervised_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.numel() == 0 or labels.numel() == 0:
            return _model_zero(self.model, self.device)
        logits = getattr(self.model, "classifier")(embeddings)
        weight = self._pseudo_class_weights(labels) if self.cfg.class_weight else None
        return F.cross_entropy(logits, labels.long(), weight=weight)

    def _pseudo_class_weights(self, labels: torch.Tensor) -> torch.Tensor:
        num_classes = self._num_classes()
        counts = torch.bincount(labels.long(), minlength=num_classes).float().clamp_min(1.0)
        return (counts.sum() / (num_classes * counts)).to(self.device)

    def _num_classes(self) -> int:
        classifier = getattr(self.model, "classifier")
        if hasattr(classifier, "out_features"):
            return int(classifier.out_features)
        return max(2, int(torch.max(self.graph.y).item()) + 1)

    def _resolve_target_idx(
        self,
        target_idx: torch.Tensor | np.ndarray | list[int] | None,
    ) -> torch.Tensor:
        if target_idx is not None:
            idx = torch.as_tensor(target_idx, device=self.device, dtype=torch.long).view(-1)
            return torch.unique(idx)

        mode = self.cfg.adapt_on.lower()
        if mode in {"test", "target"} and self.graph.test_mask is not None:
            return torch.nonzero(self.graph.test_mask, as_tuple=False).view(-1).to(self.device)
        if mode in {"val"} and self.graph.val_mask is not None:
            return torch.nonzero(self.graph.val_mask, as_tuple=False).view(-1).to(self.device)
        if mode in {"train", "source"} and self.graph.train_mask is not None:
            return torch.nonzero(self.graph.train_mask, as_tuple=False).view(-1).to(self.device)
        if mode not in {"all", "full", "test", "target", "val", "train", "source"}:
            raise ValueError("adapt_on must be one of: test, target, all, full, val, train, source.")
        return torch.arange(self.graph.num_nodes, device=self.device, dtype=torch.long)


def load_model_checkpoint(model: nn.Module, path: str | Path, *, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Load a source-pretrained or adapted checkpoint into ``model``."""

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    return checkpoint


def _use_normal_only_supervision(step: int, total_steps: int, cfg: TTAConfig) -> bool:
    """Whether the supervised objective should use normal-class nodes only.

    The final ``supervised_anomaly_tail_epochs`` steps include anomaly-class
    reliable nodes as well; earlier steps follow the normal-only settings.
    """

    tail = max(0, int(cfg.supervised_anomaly_tail_epochs))
    if tail > 0 and step > total_steps - tail:
        return False
    return bool(cfg.supervised_normal_only) or step <= int(cfg.supervised_normal_only_epochs)


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _monitor_score(metrics: dict[str, float], monitor: str) -> float:
    value = float(metrics.get(monitor, float("nan")))
    if not np.isfinite(value):
        return -float("inf")
    return -value if monitor.endswith("loss") else value


def _safe_mean(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(x.detach().float().mean().cpu().item())


def _model_zero(model: nn.Module, device: torch.device | str) -> torch.Tensor:
    for parameter in model.parameters():
        return parameter.sum() * 0.0
    return torch.tensor(0.0, device=device)


def _config_to_dict(config: TTAConfig) -> dict[str, Any]:
    return asdict(config)


def _format_tta_metrics(metrics: dict[str, float]) -> str:
    keys = [
        "epoch",
        "loss",
        "supervised_loss",
        "contrastive_loss",
        "supervised_weight",
        "reliable_count",
        "unreliable_count",
        "augmented_count",
        "target_auroc",
        "target_auprc",
        "target_f1",
    ]
    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return " | ".join(parts)
