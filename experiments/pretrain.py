"""Source pretraining: BWGNN baseline, GHRN pruning, fresh GHRN fit."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from sados.data import load_graph
from sados.models import build_detector
from sados.models.ghrn import graph_signature, prune_edges_by_pa
from sados.split import SplitConfig, read_or_create_split
from sados.trainer_pretrain import PretrainConfig, SourcePretrainer
from sados.utils import get_device, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Source pretraining for SADOS.")
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    parser.add_argument("--raw-path", default=None, help="Override data.raw_path.")
    parser.add_argument("--split", default=None, help="Override split.strategy.")
    parser.add_argument("--split-path", default=None, help="Path to a saved split npz.")
    parser.add_argument("--force-split", action="store_true", help="Rebuild split even if it exists.")
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:<id>.")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs.")
    parser.add_argument("--baseline-epochs", type=int, default=None, help="Override GHRN baseline epochs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    split_cfg_dict = dict(cfg.get("split", {}))
    model_cfg = dict(cfg.get("model", {}))
    train_cfg_dict = dict(cfg.get("train", {}))
    output_cfg = cfg.get("output", {})

    if args.split:
        split_cfg_dict["strategy"] = args.split
    if args.epochs is not None:
        train_cfg_dict["epochs"] = args.epochs

    seed = int(split_cfg_dict.get("seed", cfg.get("seed", 42)))
    set_seed(seed)

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
    out_channels = int(model_cfg.pop("out_channels", 2))
    checkpoint_dir = Path(output_cfg.get("checkpoint_dir", Path("checkpoints") / dataset_name))
    checkpoint_path = checkpoint_dir / f"{backbone}_source_{split_cfg.strategy}_seed{split_cfg.seed}.pt"
    train_cfg = PretrainConfig.from_dict(train_cfg_dict)
    device = get_device(args.device)
    original_signature = graph_signature(graph)
    log_dir = Path(output_cfg.get("log_dir", "logs/csv"))
    if backbone == "ghrn":
        ghrn_cfg = dict(cfg.get("ghrn", {}))
        baseline_epochs = int(
            args.baseline_epochs if args.baseline_epochs is not None
            else ghrn_cfg.get("baseline_epochs", 100)
        )
        baseline_path = checkpoint_dir / (
            f"bwgnn_baseline_{split_cfg.strategy}_seed{split_cfg.seed}.pt"
        )
        baseline = build_detector(
            "bwgnn", in_channels=graph.num_features,
            out_channels=out_channels, **model_cfg,
        )
        baseline_trainer = SourcePretrainer(
            baseline, graph,
            replace(train_cfg, epochs=baseline_epochs, checkpoint_path=str(baseline_path)),
            device=device,
        )
        baseline_result = baseline_trainer.fit()
        baseline_history = log_dir / (
            f"{dataset_name}_bwgnn_baseline_{split_cfg.strategy}_seed{split_cfg.seed}.csv"
        )
        baseline_trainer.save_history_csv(baseline_history, baseline_result["history"])
        baseline.eval()
        with torch.no_grad():
            probabilities = baseline(graph.x, graph.edge_index).softmax(dim=-1)
        pruned, pa_stats = prune_edges_by_pa(
            graph.edge_index, probabilities,
            delete_ratio=float(ghrn_cfg.get("delete_ratio", 0.015)),
            normalization=str(ghrn_cfg.get("normalization", "symmetric")),
            preserve_undirected=bool(ghrn_cfg.get("preserve_undirected", True)),
        )
        graph.edge_index = pruned
        print("PA pruning:", pa_stats)

    model = build_detector(
        backbone, in_channels=graph.num_features,
        out_channels=out_channels, **model_cfg,
    )
    # Save the final checkpoint with its pruned graph and source graph identity.
    trainer_cfg = replace(train_cfg, checkpoint_path=None if backbone == "ghrn" else str(checkpoint_path))
    trainer = SourcePretrainer(model, graph, trainer_cfg, device=device)
    result = trainer.fit()
    if backbone == "ghrn":
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "backbone": backbone,
            "graph_signature": original_signature,
            "pruned_edge_index": graph.edge_index.detach().cpu(),
            "pa_stats": pa_stats,
            "ghrn_config": ghrn_cfg,
            "model_config": model_cfg,
            "best_epoch": result["best_epoch"],
            "metrics": result["best_metrics"],
            "split_meta": split_result.meta,
        }, checkpoint_path)
    history_path = log_dir / f"{dataset_name}_{backbone}_source_{split_cfg.strategy}_seed{split_cfg.seed}.csv"
    trainer.save_history_csv(history_path, result["history"])

    print("Split:", split_result.meta)
    print("Best:", result["best_metrics"])
    if backbone == "ghrn":
        print("Baseline:", baseline_result["best_metrics"])
        print(f"Baseline checkpoint: {baseline_path}")
        print(f"Baseline history: {baseline_history}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
