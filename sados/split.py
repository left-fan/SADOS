"""Source/target split builders for shifted graph anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import torch

from .data import GraphData, apply_split, index_to_mask


@dataclass
class SplitConfig:
    strategy: str = "cluster"
    seed: int = 42
    num_clusters: int = 3
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    anomaly_source_ratio: float | None = None
    cluster_holdout: str = "smallest"
    target_density: str = "high"
    min_class_count_for_stratify: int = 2
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "SplitConfig":
        if cfg is None:
            return cls()
        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        values = {key: value for key, value in cfg.items() if key in known}
        values["extra"] = {key: value for key, value in cfg.items() if key not in known}
        return cls(**values)


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    source_idx: np.ndarray
    target_idx: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def attach(self, graph: GraphData) -> GraphData:
        return apply_split(graph, self.train_idx, self.val_idx, self.test_idx)

    def masks(self, num_nodes: int) -> dict[str, torch.Tensor]:
        return {
            "train_mask": index_to_mask(self.train_idx, num_nodes),
            "val_mask": index_to_mask(self.val_idx, num_nodes),
            "test_mask": index_to_mask(self.test_idx, num_nodes),
            "source_mask": index_to_mask(self.source_idx, num_nodes),
            "target_mask": index_to_mask(self.target_idx, num_nodes),
        }


def make_split(graph: GraphData, config: SplitConfig | dict[str, Any] | None = None) -> SplitResult:
    cfg = config if isinstance(config, SplitConfig) else SplitConfig.from_dict(config)
    strategy = cfg.strategy.lower().replace("_", "-")
    if strategy in {"cluster", "cluster-based", "cluster-based-split"}:
        return cluster_based_split(graph, cfg)
    if strategy in {"density", "density-based", "density-based-split"}:
        return density_based_split(graph, cfg)
    if strategy in {"random", "iid"}:
        return random_split(graph, cfg)
    raise ValueError(f"Unknown split strategy: {cfg.strategy}")


def cluster_based_split(graph: GraphData, cfg: SplitConfig) -> SplitResult:
    """Hold out one normal-node feature cluster as target-only normal data.

    This follows the paper's cluster-based protocol: k-means is applied to normal
    nodes, the smallest normal cluster is excluded from source train/validation,
    and anomalies are split randomly so source pretraining remains supervised.
    """

    x = _features_np(graph)
    y = _labels_np(graph)
    normal_idx = np.flatnonzero(y == 0)
    anomaly_idx = np.flatnonzero(y != 0)
    if normal_idx.size == 0:
        raise ValueError("Cluster split requires at least one normal node with label 0.")

    n_clusters = min(cfg.num_clusters, normal_idx.size)
    if n_clusters < 2:
        raise ValueError("Cluster split requires at least two normal nodes.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=cfg.seed, n_init=10)
    assignments = kmeans.fit_predict(x[normal_idx])
    cluster_sizes = np.bincount(assignments, minlength=n_clusters)
    holdout_cluster = _select_cluster(cluster_sizes, cfg.cluster_holdout)

    target_normals = normal_idx[assignments == holdout_cluster]
    source_normals = normal_idx[assignments != holdout_cluster]
    anomaly_source, anomaly_target = _split_anomalies(anomaly_idx, cfg)

    source_pool = np.concatenate([source_normals, anomaly_source])
    target_idx = np.concatenate([target_normals, anomaly_target])
    train_idx, val_idx = _train_val_split(source_pool, y, cfg)

    meta = {
        "strategy": "cluster",
        "seed": cfg.seed,
        "num_clusters": int(n_clusters),
        "cluster_sizes": cluster_sizes.astype(int).tolist(),
        "holdout_cluster": int(holdout_cluster),
        "cluster_holdout": cfg.cluster_holdout,
        "target_normal_count": int(target_normals.size),
        "target_anomaly_count": int(anomaly_target.size),
    }
    return _finalize_split(train_idx, val_idx, target_idx, source_pool, target_idx, meta)


def density_based_split(graph: GraphData, cfg: SplitConfig) -> SplitResult:
    """Partition the undirected graph with METIS and hold out one density side."""

    try:
        import pymetis
    except ImportError as exc:
        raise ImportError("Density split requires PyMetis: pip install pymetis") from exc

    if graph.num_nodes < 2:
        raise ValueError("Density split needs at least two nodes.")
    if cfg.target_density not in {"high", "low"}:
        raise ValueError("target_density must be 'high' or 'low'.")

    y = _labels_np(graph)
    edges = graph.edge_index.detach().cpu().numpy()
    adjacency = [set() for _ in range(graph.num_nodes)]
    for src, dst in zip(edges[0], edges[1]):
        a, b = int(src), int(dst)
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    degree = np.asarray([len(neighbors) for neighbors in adjacency], dtype=np.float64)
    _, membership = pymetis.part_graph(
        2,
        adjacency=[sorted(neighbors) for neighbors in adjacency],
        options=pymetis.Options(seed=cfg.seed),
    )
    assignments = np.asarray(membership, dtype=np.int64)
    means = np.asarray([degree[assignments == i].mean() for i in range(2)])
    target_cluster = int(np.argmax(means) if cfg.target_density == "high" else np.argmin(means))

    target_idx = np.flatnonzero(assignments == target_cluster)
    source_pool = np.flatnonzero(assignments != target_cluster)
    train_idx, val_idx = _train_val_split(source_pool, y, cfg)

    meta = {
        "strategy": "density",
        "seed": cfg.seed,
        "partitioner": "METIS",
        "num_clusters": 2,
        "cluster_mean_degrees": means.tolist(),
        "target_cluster": int(target_cluster),
        "target_density": cfg.target_density,
    }
    return _finalize_split(train_idx, val_idx, target_idx, source_pool, target_idx, meta)


def random_split(graph: GraphData, cfg: SplitConfig) -> SplitResult:
    y = _labels_np(graph)
    all_idx = np.arange(graph.num_nodes)
    train_idx, rest_idx = _safe_train_test_split(
        all_idx,
        y[all_idx],
        train_size=cfg.train_ratio,
        seed=cfg.seed,
        min_class_count=cfg.min_class_count_for_stratify,
    )
    val_fraction = cfg.val_ratio / max(1e-12, 1.0 - cfg.train_ratio)
    val_idx, test_idx = _safe_train_test_split(
        rest_idx,
        y[rest_idx],
        train_size=val_fraction,
        seed=cfg.seed + 1,
        min_class_count=cfg.min_class_count_for_stratify,
    )
    meta = {"strategy": "random", "seed": cfg.seed}
    source_idx = np.concatenate([train_idx, val_idx])
    return _finalize_split(train_idx, val_idx, test_idx, source_idx, test_idx, meta)


def save_split(result: SplitResult, path: str | Path, num_nodes: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    masks = result.masks(num_nodes)
    np.savez_compressed(
        path,
        train_idx=result.train_idx,
        val_idx=result.val_idx,
        test_idx=result.test_idx,
        source_idx=result.source_idx,
        target_idx=result.target_idx,
        train_mask=masks["train_mask"].numpy(),
        val_mask=masks["val_mask"].numpy(),
        test_mask=masks["test_mask"].numpy(),
        meta=json.dumps(result.meta, sort_keys=True),
    )


def load_split(path: str | Path) -> SplitResult:
    payload = np.load(path, allow_pickle=True)
    meta_value = payload["meta"].item() if "meta" in payload else "{}"
    return SplitResult(
        train_idx=payload["train_idx"].astype(np.int64),
        val_idx=payload["val_idx"].astype(np.int64),
        test_idx=payload["test_idx"].astype(np.int64),
        source_idx=payload["source_idx"].astype(np.int64),
        target_idx=payload["target_idx"].astype(np.int64),
        meta=json.loads(meta_value),
    )


def read_or_create_split(
    graph: GraphData,
    path: str | Path,
    config: SplitConfig | dict[str, Any] | None = None,
    *,
    force: bool = False,
) -> SplitResult:
    path = Path(path)
    if path.exists() and not force:
        saved = load_split(path)
        strategy = (config.strategy if isinstance(config, SplitConfig) else (config or {}).get("strategy", "cluster"))
        if strategy.lower().startswith("density") and saved.meta.get("partitioner") != "METIS":
            force = True  # Replace older degree-k-means proxy splits.
        else:
            return saved
    result = make_split(graph, config)
    save_split(result, path, graph.num_nodes)
    return result


def _finalize_split(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    meta: dict[str, Any],
) -> SplitResult:
    train_idx = np.unique(train_idx.astype(np.int64))
    val_idx = np.unique(val_idx.astype(np.int64))
    test_idx = np.unique(test_idx.astype(np.int64))
    source_idx = np.unique(source_idx.astype(np.int64))
    target_idx = np.unique(target_idx.astype(np.int64))
    meta.update(
        {
            "train_count": int(train_idx.size),
            "val_count": int(val_idx.size),
            "test_count": int(test_idx.size),
            "source_count": int(source_idx.size),
            "target_count": int(target_idx.size),
        }
    )
    return SplitResult(train_idx, val_idx, test_idx, source_idx, target_idx, meta)


def _features_np(graph: GraphData) -> np.ndarray:
    return graph.x.detach().cpu().numpy()


def _labels_np(graph: GraphData) -> np.ndarray:
    return graph.y.detach().cpu().numpy().reshape(-1)


def _degree_np(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    edge_index = edge_index.detach().cpu().long()
    degree = torch.bincount(edge_index[0], minlength=num_nodes)
    degree += torch.bincount(edge_index[1], minlength=num_nodes)
    return degree.numpy().astype(np.float64)


def _select_cluster(cluster_sizes: np.ndarray, mode: str) -> int:
    if mode == "smallest":
        return int(np.argmin(cluster_sizes))
    if mode == "largest":
        return int(np.argmax(cluster_sizes))
    if mode.startswith("id:"):
        return int(mode.split(":", 1)[1])
    raise ValueError("cluster_holdout must be 'smallest', 'largest', or 'id:<int>'.")


def _split_anomalies(anomaly_idx: np.ndarray, cfg: SplitConfig) -> tuple[np.ndarray, np.ndarray]:
    if anomaly_idx.size == 0:
        return anomaly_idx, anomaly_idx
    source_ratio = cfg.anomaly_source_ratio
    if source_ratio is None:
        source_ratio = min(0.95, cfg.train_ratio + cfg.val_ratio)
    rng = np.random.default_rng(cfg.seed)
    shuffled = rng.permutation(anomaly_idx)
    cut = int(round(shuffled.size * source_ratio))
    if shuffled.size > 1:
        cut = min(max(cut, 1), shuffled.size - 1)
    else:
        cut = 1
    return shuffled[:cut], shuffled[cut:]


def _train_val_split(source_pool: np.ndarray, y: np.ndarray, cfg: SplitConfig) -> tuple[np.ndarray, np.ndarray]:
    if source_pool.size == 0:
        raise ValueError("Source pool is empty; cannot create train/validation split.")
    source_train_ratio = cfg.train_ratio / max(1e-12, cfg.train_ratio + cfg.val_ratio)
    return _safe_train_test_split(
        source_pool,
        y[source_pool],
        train_size=source_train_ratio,
        seed=cfg.seed,
        min_class_count=cfg.min_class_count_for_stratify,
    )


def _safe_train_test_split(
    idx: np.ndarray,
    labels: np.ndarray,
    *,
    train_size: float,
    seed: int,
    min_class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size <= 1:
        return idx, np.asarray([], dtype=np.int64)
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    stratify = labels if unique.size > 1 and counts.min() >= min_class_count else None
    try:
        train_idx, test_idx = train_test_split(
            idx,
            train_size=train_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(idx)
        cut = int(round(shuffled.size * train_size))
        cut = min(max(cut, 1), shuffled.size - 1)
        train_idx, test_idx = shuffled[:cut], shuffled[cut:]
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)
