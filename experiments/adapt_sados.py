"""Run SADOS test-time adaptation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from sados.data import GraphData, load_graph
from sados.models import build_detector
from sados.models.ghrn import graph_signature
from sados.split import SplitConfig, read_or_create_split
from sados.trainer_tta import SadosTTATrainer, TTAConfig, load_model_checkpoint
from sados.utils import get_device, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SADOS test-time adaptation.")
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    parser.add_argument("--raw-path", default=None, help="Override data.raw_path.")
    parser.add_argument("--split", default=None, help="Override split.strategy.")
    parser.add_argument("--split-path", default=None, help="Path to a saved split npz.")
    parser.add_argument("--force-split", action="store_true", help="Rebuild split even if it exists.")
    parser.add_argument("--checkpoint", default=None, help="Source-pretrained checkpoint path.")
    parser.add_argument("--output-checkpoint", default=None, help="Adapted checkpoint output path.")
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:<id>.")
    parser.add_argument("--epochs", type=int, default=None, help="Override tta.epochs.")
    parser.add_argument("--adapt-on", default=None, help="test, all, val, train/source.")
    parser.add_argument("--tta-mode", default=None, help="single or stream.")
    parser.add_argument("--stream-batch-size", type=int, default=None, help="Target nodes per streaming TTT step.")
    parser.add_argument("--stream-shuffle", action="store_true", help="Shuffle target stream order before batching.")
    parser.add_argument("--updates-per-batch", type=int, default=None, help="Gradient updates per arrived stream batch.")
    return parser.parse_args()


def _peak_metrics(
    history: list[dict],
    keys: tuple[str, ...],
) -> dict[str, tuple[float, int]]:
    """Highest value of each metric across TTA steps and the step it occurred at."""

    peaks: dict[str, tuple[float, int]] = {}
    for key in keys:
        best_value = float("nan")
        best_epoch = -1
        for row in history:
            value = float(row.get(key, float("nan")))
            if math.isfinite(value) and (not math.isfinite(best_value) or value > best_value):
                best_value = value
                best_epoch = int(row.get("epoch", -1))
        peaks[key] = (best_value, best_epoch)
    return peaks


def _report_best_and_final(
    result: dict,
    trainer: SadosTTATrainer,
    graph: GraphData,
    threshold: float,
) -> None:
    """Print peak training metrics and final per-class target metrics."""

    keys = ("target_auroc", "target_auprc", "target_f1")
    peaks = _peak_metrics(result.get("history", []), keys)
    print("Best (peak) during TTA:")
    for key in keys:
        value, epoch = peaks[key]
        print(f"  {key:<14}= {value:.4f} @ epoch {epoch}")

    idx = trainer.target_idx
    y_true = graph.y[idx].long()
    score = trainer.anomaly_scores()[idx]
    pred = (score >= float(threshold)).long()
    normal_mask = y_true == 0
    anomaly_mask = y_true != 0

    def _class_accuracy(mask: torch.Tensor, label: int) -> tuple[int, int, float]:
        total = int(mask.sum().item())
        if total == 0:
            return 0, 0, float("nan")
        correct = int((pred[mask] == label).sum().item())
        return total, correct, correct / total

    n_total, n_correct, n_acc = _class_accuracy(normal_mask, 0)
    a_total, a_correct, a_acc = _class_accuracy(anomaly_mask, 1)
    overall = trainer.evaluate(idx)
    print(f"Final model on target (n={int(idx.numel())}):")
    print(
        f"  overall  AUROC={overall['auroc']:.4f}  AUPRC={overall['auprc']:.4f}  "
        f"F1={overall['f1']:.4f}"
    )
    print(f"  normal  ({n_total:5d}): correct={n_correct:5d}  accuracy={n_acc:.4f}")
    print(f"  anomaly ({a_total:5d}): correct={a_correct:5d}  accuracy={a_acc:.4f}")


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
    if args.epochs is not None:
        tta_cfg_dict["epochs"] = args.epochs
    if args.adapt_on:
        tta_cfg_dict["adapt_on"] = args.adapt_on
    if args.tta_mode:
        tta_cfg_dict["mode"] = args.tta_mode
    if args.stream_batch_size is not None:
        tta_cfg_dict["stream_batch_size"] = args.stream_batch_size
    if args.stream_shuffle:
        tta_cfg_dict["stream_shuffle"] = True
    if args.updates_per_batch is not None:
        tta_cfg_dict["updates_per_batch"] = args.updates_per_batch

    seed = int(split_cfg_dict.get("seed", cfg.get("seed", 42)))
    set_seed(seed)
    device = get_device(args.device)

    dataset_name = data_cfg.get("name", Path(args.config).stem)
    raw_path = args.raw_path or data_cfg.get("raw_path", Path("data/raw") / dataset_name)
    graph = load_graph(
        raw_path,
        name=dataset_name,
        standardize_features=bool(data_cfg.get("standardize_features", True)),
        make_undirected=bool(data_cfg.get("make_undirected", True)),
        remove_self_loops=bool(data_cfg.get("remove_self_loops", False)),
    )

    split_cfg = SplitConfig.from_dict(split_cfg_dict)
    split_path = args.split_path or Path("data/splits") / (
        f"{dataset_name}_{split_cfg.strategy}_seed{split_cfg.seed}.npz"
    )
    split_result = read_or_create_split(
        graph,
        split_path,
        split_cfg,
        force=args.force_split,
    )
    split_result.attach(graph)

    backbone = model_cfg.pop("backbone", "ghrn").lower()
    model = build_detector(
        backbone,
        in_channels=graph.num_features,
        out_channels=int(model_cfg.pop("out_channels", 2)),
        **model_cfg,
    )

    checkpoint_dir = Path(output_cfg.get("checkpoint_dir", Path("checkpoints") / dataset_name))
    source_checkpoint = args.checkpoint or checkpoint_dir / (
        f"{backbone}_source_{split_cfg.strategy}_seed{split_cfg.seed}.pt"
    )
    checkpoint = load_model_checkpoint(model, source_checkpoint, device=device)
    if backbone == "ghrn":
        if checkpoint.get("backbone") != "ghrn":
            raise ValueError("Source checkpoint is not a GHRN checkpoint.")
        if graph_signature(graph) != checkpoint.get("graph_signature"):
            raise ValueError("Source graph or split differs from the GHRN checkpoint.")
        pruned_edges = checkpoint.get("pruned_edge_index")
        if not isinstance(pruned_edges, torch.Tensor) or pruned_edges.ndim != 2 or pruned_edges.size(0) != 2:
            raise ValueError("GHRN checkpoint is missing its pruned graph.")
        graph.edge_index = pruned_edges.long().cpu()

    adapted_checkpoint = args.output_checkpoint or checkpoint_dir / (
        f"sados_{backbone}_tta_{split_cfg.strategy}_seed{split_cfg.seed}.pt"
    )
    tta_cfg_dict.setdefault("checkpoint_path", str(adapted_checkpoint))
    tta_cfg = TTAConfig.from_dict(tta_cfg_dict)

    trainer = SadosTTATrainer(model, graph, tta_cfg, device=device)
    result = trainer.fit()

    log_dir = Path(output_cfg.get("log_dir", "logs/csv"))
    history_path = log_dir / f"{dataset_name}_sados_{backbone}_tta_{split_cfg.strategy}_seed{split_cfg.seed}.csv"
    trainer.save_history_csv(history_path, result["history"])

    final_metrics = result["history"][-1] if result["history"] else {}
    print("Split:", split_result.meta)
    _report_best_and_final(result, trainer, graph, tta_cfg.threshold)
    if result["best_metrics"]:
        print("Best:", result["best_metrics"])
    else:
        print("Best: disabled (strict TTT mode; no target-label model selection)")
    print("Final:", final_metrics)
    print(f"Source checkpoint: {source_checkpoint}")
    print(f"Adapted checkpoint: {adapted_checkpoint}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
