"""Data loading utilities for node-attributed graph anomaly datasets.

The loaders intentionally accept several common research formats because GAD
benchmarks are distributed inconsistently across papers: PyG ``.pt`` objects,
dictionary ``.pkl`` files, sparse/dense ``.npz`` files, and MATLAB ``.mat``
files are all supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import torch


SUPPORTED_EXTENSIONS = (".pt", ".pth", ".npz", ".mat", ".pkl", ".pickle")


@dataclass
class GraphData:
    """In-memory graph container used by SADOS.

    Attributes:
        x: Node features with shape ``[num_nodes, num_features]``.
        edge_index: COO edges with shape ``[2, num_edges]``.
        y: Binary anomaly labels where 0 is normal and 1 is anomalous.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    name: str = "graph"
    train_mask: torch.Tensor | None = None
    val_mask: torch.Tensor | None = None
    test_mask: torch.Tensor | None = None
    aux: dict[str, Any] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))

    @property
    def num_features(self) -> int:
        return int(self.x.size(1))

    @property
    def num_classes(self) -> int:
        return int(torch.unique(self.y).numel())

    def to(self, device: torch.device | str) -> "GraphData":
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.y = self.y.to(device)
        if self.train_mask is not None:
            self.train_mask = self.train_mask.to(device)
        if self.val_mask is not None:
            self.val_mask = self.val_mask.to(device)
        if self.test_mask is not None:
            self.test_mask = self.test_mask.to(device)
        return self


def load_graph(
    path_or_name: str | Path,
    root: str | Path = "data/raw",
    *,
    name: str | None = None,
    standardize_features: bool = True,
    make_undirected: bool = True,
    remove_self_loops: bool = False,
) -> GraphData:
    """Load a graph from a file, dataset directory, or dataset name."""

    path = resolve_data_path(path_or_name, root)
    payload = _load_payload(path)
    graph = _graph_from_payload(payload, name=name or _infer_name(path))

    if standardize_features:
        graph.x = standardize_feature_tensor(graph.x)
    if remove_self_loops:
        graph.edge_index = remove_self_loop_edges(graph.edge_index)
    if make_undirected:
        graph.edge_index = to_undirected(graph.edge_index)

    validate_graph(graph)
    graph.aux.setdefault("source_path", str(path))
    return graph


def resolve_data_path(path_or_name: str | Path, root: str | Path = "data/raw") -> Path:
    """Resolve a dataset name or path to an existing graph file."""

    candidate = Path(path_or_name)
    if candidate.exists():
        if candidate.is_file():
            return candidate
        return _find_graph_file(candidate)

    rooted = Path(root) / candidate
    if rooted.exists():
        if rooted.is_file():
            return rooted
        return _find_graph_file(rooted)

    raise FileNotFoundError(
        f"Could not find dataset '{path_or_name}'. Put a supported file under "
        f"{Path(root) / candidate} or pass an explicit file path."
    )


def apply_split(
    graph: GraphData,
    train_idx: Iterable[int],
    val_idx: Iterable[int],
    test_idx: Iterable[int],
) -> GraphData:
    """Attach split masks to a graph and return it."""

    graph.train_mask = index_to_mask(train_idx, graph.num_nodes)
    graph.val_mask = index_to_mask(val_idx, graph.num_nodes)
    graph.test_mask = index_to_mask(test_idx, graph.num_nodes)
    return graph


def index_to_mask(indices: Iterable[int], num_nodes: int) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    idx = torch.as_tensor(np.asarray(list(indices), dtype=np.int64), dtype=torch.long)
    if idx.numel() > 0:
        mask[idx] = True
    return mask


def save_processed_graph(graph: GraphData, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "x": graph.x.cpu(),
            "edge_index": graph.edge_index.cpu(),
            "y": graph.y.cpu(),
            "name": graph.name,
            "aux": graph.aux,
        },
        path,
    )


def standardize_feature_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-12)
    return (x - mean) / std


def to_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    rev = edge_index.flip(0)
    merged = torch.cat([edge_index, rev], dim=1)
    return unique_edges(merged)


def remove_self_loop_edges(edge_index: torch.Tensor) -> torch.Tensor:
    keep = edge_index[0] != edge_index[1]
    return edge_index[:, keep]


def unique_edges(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index
    edge_index = edge_index.long()
    num_nodes = int(edge_index.max().item()) + 1
    keys = edge_index[0] * num_nodes + edge_index[1]
    order = torch.argsort(keys)
    sorted_edges = edge_index[:, order]
    sorted_keys = keys[order]
    keep = torch.ones(sorted_keys.numel(), dtype=torch.bool)
    keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return sorted_edges[:, keep]


def validate_graph(graph: GraphData) -> None:
    if graph.x.dim() != 2:
        raise ValueError("Node features must be a 2D tensor.")
    if graph.edge_index.dim() != 2 or graph.edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if graph.y.dim() != 1:
        raise ValueError("Labels must be a 1D tensor.")
    if graph.x.size(0) != graph.y.size(0):
        raise ValueError("Feature and label node counts do not match.")
    if graph.edge_index.numel() > 0:
        min_id = int(graph.edge_index.min().item())
        max_id = int(graph.edge_index.max().item())
        if min_id < 0 or max_id >= graph.num_nodes:
            raise ValueError("edge_index contains node ids outside [0, num_nodes).")


def _find_graph_file(directory: Path) -> Path:
    for ext in SUPPORTED_EXTENSIONS:
        files = sorted(directory.glob(f"*{ext}"))
        if files:
            return files[0]
    raise FileNotFoundError(
        f"No supported graph file found in {directory}. Supported: {SUPPORTED_EXTENSIONS}"
    )


def _load_payload(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    if suffix == ".npz":
        return _load_npz(path)
    if suffix == ".mat":
        return loadmat(path)
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported data extension: {path.suffix}")


def _load_npz(path: Path) -> dict[str, Any]:
    npz = np.load(path, allow_pickle=True)
    payload: dict[str, Any] = {key: npz[key] for key in npz.files}
    sparse_prefixes = ("adj", "adjacency", "A", "network", "Network")
    for prefix in sparse_prefixes:
        data_key = f"{prefix}_data"
        if {data_key, f"{prefix}_indices", f"{prefix}_indptr", f"{prefix}_shape"}.issubset(payload):
            payload[prefix] = sp.csr_matrix(
                (
                    payload[data_key],
                    payload[f"{prefix}_indices"],
                    payload[f"{prefix}_indptr"],
                ),
                shape=tuple(payload[f"{prefix}_shape"]),
            )
    return payload


def _graph_from_payload(payload: Any, *, name: str) -> GraphData:
    if isinstance(payload, (list, tuple)):
        if not payload:
            raise ValueError("Loaded payload is empty.")
        payload = payload[0]

    if isinstance(payload, Mapping):
        return _graph_from_mapping(payload, name=name)

    return _graph_from_object(payload, name=name)


def _graph_from_mapping(mapping: Mapping[str, Any], *, name: str) -> GraphData:
    x_value = _pick(mapping, ("x", "features", "feature", "feat", "attrs", "Attributes"))
    y_value = _pick(mapping, ("y", "label", "labels", "target", "targets", "gnd", "Class"))
    edge_key, edge_value = _pick_with_key(
        mapping,
        (
            "edge_index",
            "edges",
            "edge_list",
            "homo",
            "adj",
            "adjacency",
            "A",
            "network",
            "Network",
        ),
    )
    if x_value is None or y_value is None or edge_value is None:
        raise KeyError(
            "Could not find graph fields. Expected feature keys like x/features, "
            "label keys like y/labels, and edge keys like edge_index/adj."
        )
    x = _coerce_features(x_value)
    y = _coerce_labels(y_value)
    edge_index = _coerce_edge_index(edge_value, assume_adjacency=edge_key in _ADJ_KEYS)
    return GraphData(x=x, edge_index=edge_index, y=y, name=name)


def _graph_from_object(obj: Any, *, name: str) -> GraphData:
    fields = {}
    for key in ("x", "features", "edge_index", "edge_attr", "y", "labels", "adj"):
        if hasattr(obj, key):
            fields[key] = getattr(obj, key)
    if hasattr(obj, "num_nodes"):
        fields["num_nodes"] = getattr(obj, "num_nodes")
    return _graph_from_mapping(fields, name=name)


def _coerce_features(value: Any) -> torch.Tensor:
    if sp.issparse(value):
        value = value.toarray()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().float()
    else:
        arr = np.asarray(value)
        if arr.dtype == object and arr.size == 1:
            arr = np.asarray(arr.item())
        tensor = torch.as_tensor(arr, dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(1)
    return tensor.contiguous()


def _coerce_labels(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().long()
    else:
        arr = np.asarray(value)
        if arr.dtype == object and arr.size == 1:
            arr = np.asarray(arr.item())
        tensor = torch.as_tensor(arr, dtype=torch.long)
    return tensor.view(-1).contiguous()


_ADJ_KEYS = {"homo", "adj", "adjacency", "A", "network", "Network"}


def _coerce_edge_index(value: Any, *, assume_adjacency: bool = False) -> torch.Tensor:
    if sp.issparse(value):
        coo = value.tocoo()
        return torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)

    if isinstance(value, torch.Tensor):
        if value.layout == torch.sparse_coo:
            value = value.coalesce().indices()
        else:
            value = value.detach().cpu()
        edge_index = value.long()
    else:
        arr = np.asarray(value)
        if arr.dtype == object and arr.size == 1:
            arr = np.asarray(arr.item())
        if assume_adjacency and arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            row, col = np.nonzero(arr)
            arr = np.vstack([row, col])
        edge_index = torch.as_tensor(arr, dtype=torch.long)

    if edge_index.dim() != 2:
        raise ValueError("Edges must be a matrix-like object.")
    if edge_index.size(0) != 2 and edge_index.size(1) == 2:
        edge_index = edge_index.t().contiguous()
    if edge_index.size(0) != 2:
        raise ValueError("Could not coerce edges to shape [2, num_edges].")
    return edge_index.contiguous()


def _pick(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _pick_with_key(mapping: Mapping[str, Any], keys: Iterable[str]) -> tuple[str | None, Any]:
    for key in keys:
        if key in mapping:
            return key, mapping[key]
    return None, None


def _infer_name(path: Path) -> str:
    if path.parent.name != "raw":
        return path.parent.name
    return path.stem
