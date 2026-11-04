"""Report pseudo-label accuracy on the SADOS reliable set.

Read-only diagnostic: loads a source (or adapted) checkpoint, recomputes the
confidence partition on the target nodes, and reports how accurate the
reliable-set pseudo-labels are for truly normal vs truly anomalous nodes.
Ground-truth labels are used only for reporting, never for training or model
selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sados.data import load_graph
from sados.models import build_detector
from sados.partition import partition_by_confidence
from sados.split import SplitConfig, read_or_create_split
from sados.trainer_tta import TTAConfig, load_model_checkpoint
from sados.utils import get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reliable-set pseudo-label accuracy.")
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint (default: source).")
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:<id>.")
    parser.add_argument("--split", default=None, help="Override split.strategy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    split_cfg_dict = dict(cfg.get("split", {}))
    model_cfg = dict(cfg.get("model", {}))
    tta_cfg_dict = dict(cfg.get("tta", {}))
    output_cfg = cfg.get("output", {})
    if args.split:
        split_cfg_dict["strategy"] = args.split

    dataset_name = data_cfg.get("name", Path(args.config).stem)
    raw_path = data_cfg.get("raw_path", Path("data/raw") / dataset_name)
    graph = load_graph(
        raw_path,
        name=dataset_name,
        standardize_features=bool(data_cfg.get("standardize_features", True)),
        make_undirected=bool(data_cfg.get("make_undirected", True)),
        remove_self_loops=bool(data_cfg.get("remove_self_loops", False)),
    )
    split_cfg = SplitConfig.from_dict(split_cfg_dict)
    split_path = Path("data/splits") / f"{dataset_name}_{split_cfg.strategy}_seed{split_cfg.seed}.npz"
    split_result = read_or_create_split(graph, split_path, split_cfg)
    split_result.attach(graph)

    backbone = model_cfg.pop("backbone", "ghrn").lower()
    out_channels = int(model_cfg.pop("out_channels", 2))
    device = get_device(args.device)
    checkpoint_dir = Path(output_cfg.get("checkpoint_dir", Path("checkpoints") / dataset_name))
    source_checkpoint = args.checkpoint or checkpoint_dir / (
        f"{backbone}_source_{split_cfg.strategy}_seed{split_cfg.seed}.pt"
    )
    model = build_detector(
        backbone,
        in_channels=graph.num_features,
        out_channels=out_channels,
        **model_cfg,
    )
    checkpoint = load_model_checkpoint(model, source_checkpoint, device=device)
    model.to(device)
    pruned = checkpoint.get("pruned_edge_index")
    if backbone == "ghrn" and isinstance(pruned, torch.Tensor):
        graph.edge_index = pruned.long().cpu()
    graph.to(device)

    model.eval()
    with torch.no_grad():
        logits, embeddings = model(
            graph.x,
            graph.edge_index,
            return_embedding=True,
        )

    target_idx = torch.nonzero(graph.test_mask, as_tuple=False).view(-1).to(device)
    tta_cfg = TTAConfig.from_dict(tta_cfg_dict)
    partition = partition_by_confidence(
        logits=logits.detach(),
        embeddings=embeddings.detach(),
        edge_index=graph.edge_index,
        node_idx=target_idx,
        config=tta_cfg.partition,
    )

    normal_label = int(tta_cfg.augmentation.normal_label)
    anomaly_label = 1 - normal_label
    reliable_idx = partition.reliable_idx
    y_true = graph.y[reliable_idx].long()
    y_pred = partition.pseudo_labels[reliable_idx].long()
    normal_mask = y_true == normal_label
    anomaly_mask = y_true != normal_label

    def _accuracy(mask: torch.Tensor, pred_label: int) -> tuple[int, int, float]:
        total = int(mask.sum().item())
        if total == 0:
            return 0, 0, float("nan")
        correct = int((y_pred[mask] == pred_label).sum().item())
        return total, correct, correct / total

    n_total, n_correct, n_acc = _accuracy(normal_mask, normal_label)
    a_total, a_correct, a_acc = _accuracy(anomaly_mask, anomaly_label)
    top1 = int((y_pred == y_true).sum().item())
    overall = top1 / reliable_idx.numel() if reliable_idx.numel() else float("nan")

    print(f"Dataset: {dataset_name} | backbone: {backbone} | checkpoint: {source_checkpoint}")
    print(f"Target nodes: {int(target_idx.numel())} | reliable: {int(reliable_idx.numel())}")
    print()
    print("Class            total   correct   accuracy")
    print(f"normal ({normal_label})       {n_total:6d}  {n_correct:8d}   {n_acc:8.4f}")
    print(f"anomaly ({anomaly_label})      {a_total:6d}  {a_correct:8d}   {a_acc:8.4f}")
    print(f"overall          {int(reliable_idx.numel()):6d}  {top1:8d}   {overall:8.4f}")
    print()
    print(
        "pseudo distribution among reliable: "
        f"normal={(y_pred == normal_label).sum().item()}, "
        f"anomaly={(y_pred == anomaly_label).sum().item()}"
    )
    confidence = partition.calibrated_confidence[reliable_idx]
    if confidence.numel():
        print(f"mean calibrated confidence: {float(confidence.mean().item()):.4f}")


if __name__ == "__main__":
    main()
