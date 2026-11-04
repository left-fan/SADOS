"""Supervised source-domain pretraining for graph anomaly detectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import csv
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import GraphData
from .metrics import binary_classification_metrics


@dataclass
class PretrainConfig:
    epochs: int = 200
    lr: float = 0.005
    weight_decay: float = 5e-5
    patience: int = 50
    class_weight: bool = True
    monitor: str = "val_auroc"
    threshold: float = 0.5
    clip_grad_norm: float | None = None
    log_every: int = 10
    checkpoint_path: str | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "PretrainConfig":
        if cfg is None:
            return cls()
        known = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in cfg.items() if key in known})


class SourcePretrainer:
    def __init__(
        self,
        model: nn.Module,
        graph: GraphData,
        config: PretrainConfig | dict[str, Any] | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.graph = graph.to(device)
        self.cfg = config if isinstance(config, PretrainConfig) else PretrainConfig.from_dict(config)
        self.device = torch.device(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss(weight=self._class_weights())

    def fit(self) -> dict[str, Any]:
        self._check_masks()
        best_score = -float("inf")
        best_epoch = -1
        best_state = copy.deepcopy(self.model.state_dict())
        best_metrics: dict[str, Any] = {}
        history: list[dict[str, float]] = []

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_epoch()
            metrics = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                **self.evaluate_all(),
            }
            history.append(metrics)
            score = _monitor_score(metrics, self.cfg.monitor)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_metrics = metrics
                best_state = copy.deepcopy(self.model.state_dict())
                if self.cfg.checkpoint_path:
                    self.save_checkpoint(self.cfg.checkpoint_path, epoch, metrics)

            if self.cfg.log_every and epoch % self.cfg.log_every == 0:
                print(_format_metrics(metrics))

            if epoch - best_epoch >= self.cfg.patience:
                break

        self.model.load_state_dict(best_state)
        return {
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_metrics": best_metrics,
            "history": history,
        }

    def train_epoch(self) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(self.graph.x, self.graph.edge_index)
        loss = self.criterion(logits[self.graph.train_mask], self.graph.y[self.graph.train_mask])
        loss.backward()
        if self.cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    @torch.no_grad()
    def evaluate_all(self) -> dict[str, float]:
        return {
            **_prefix_metrics("train", self.evaluate(self.graph.train_mask)),
            **_prefix_metrics("val", self.evaluate(self.graph.val_mask)),
            **_prefix_metrics("test", self.evaluate(self.graph.test_mask)),
        }

    @torch.no_grad()
    def evaluate(self, mask: torch.Tensor | None) -> dict[str, float]:
        if mask is None or int(mask.sum().item()) == 0:
            return {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan")}
        self.model.eval()
        logits = self.model(self.graph.x, self.graph.edge_index)
        prob = torch.softmax(logits, dim=-1)
        score = prob[:, 1] if prob.size(-1) > 1 else prob.squeeze(-1)
        y_true = self.graph.y[mask].detach().cpu().numpy()
        y_score = score[mask].detach().cpu().numpy()
        return binary_classification_metrics(
            y_true,
            y_score,
            threshold=self.cfg.threshold,
        )

    def save_checkpoint(self, path: str | Path, epoch: int, metrics: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.cfg.__dict__,
                "graph": {
                    "name": self.graph.name,
                    "num_nodes": self.graph.num_nodes,
                    "num_edges": self.graph.num_edges,
                    "num_features": self.graph.num_features,
                    "num_classes": self.graph.num_classes,
                },
            },
            path,
        )

    def save_history_csv(self, path: str | Path, history: list[dict[str, float]]) -> None:
        if not history:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    def _class_weights(self) -> torch.Tensor | None:
        if not self.cfg.class_weight:
            return None
        train_y = self.graph.y[self.graph.train_mask]
        num_classes = int(torch.max(self.graph.y).item()) + 1
        counts = torch.bincount(train_y, minlength=num_classes).float()
        counts = counts.clamp_min(1.0)
        weights = counts.sum() / (num_classes * counts)
        return weights.to(self.device)

    def _check_masks(self) -> None:
        if self.graph.train_mask is None or self.graph.val_mask is None:
            raise ValueError("Graph must have train_mask and val_mask before pretraining.")
        if int(self.graph.train_mask.sum().item()) == 0:
            raise ValueError("train_mask is empty.")
        train_classes = torch.unique(self.graph.y[self.graph.train_mask])
        if train_classes.numel() < 2:
            raise ValueError(
                "Source pretraining needs both normal and anomaly labels in train_mask."
            )


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _monitor_score(metrics: dict[str, float], monitor: str) -> float:
    value = metrics.get(monitor, float("nan"))
    if np.isnan(value):
        if monitor.endswith("loss"):
            return -float("inf")
        return -float("inf")
    return -float(value) if monitor.endswith("loss") else float(value)


def _format_metrics(metrics: dict[str, float]) -> str:
    keys = ["epoch", "train_loss", "val_auroc", "val_auprc", "val_f1", "test_auroc"]
    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return " | ".join(parts)
