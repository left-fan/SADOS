"""Detector factory."""

from __future__ import annotations

from typing import Any

from .gcn import GCNDetector
from .bwgnn import BWGNNDetector
from .ghrn import GHRNDetector


def build_detector(
    backbone: str,
    *,
    in_channels: int,
    hidden_channels: int = 64,
    out_channels: int = 2,
    num_layers: int = 2,
    dropout: float = 0.5,
    **_: Any,
):
    name = backbone.lower()
    if name == "gcn":
        return GCNDetector(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    if name in {"bwgnn", "ghrn"}:
        model_class = BWGNNDetector if name == "bwgnn" else GHRNDetector
        return model_class(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown backbone: {backbone}")
