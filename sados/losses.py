"""Loss helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_ce_loss(logits: torch.Tensor, labels: torch.Tensor, weight=None) -> torch.Tensor:
    return F.cross_entropy(logits, labels, weight=weight)
