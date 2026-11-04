"""Convert the DGL-serialized reddit graph into SADOS's ``GraphData`` format.

The reddit graph ships as a DGL binary file (loadable with ``dgl.load_graphs``).
``sados.data.load_graph`` does not understand that format, so this entrypoint
reads the graph with DGL and writes the project's unified ``.pt`` container
(``x``, ``edge_index``, ``y``) that the rest of the pipeline consumes.

Usage:
    python -m experiments.convert_reddit
    python -m experiments.convert_reddit --input data/raw/reddit --output data/raw/reddit.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dgl
import torch

from sados.data import GraphData, save_processed_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the DGL reddit graph to SADOS .pt format.")
    parser.add_argument("--input", default="data/raw/reddit", help="Path to the DGL-serialized graph.")
    parser.add_argument("--output", default="data/raw/reddit.pt", help="Output .pt path.")
    parser.add_argument("--name", default="reddit", help="Dataset name stored in the output.")
    return parser.parse_args()


def convert_reddit(input_path: str | Path, output_path: str | Path, *, name: str = "reddit") -> GraphData:
    graphs, _ = dgl.load_graphs(str(input_path))
    if not graphs:
        raise ValueError(f"No graph found in {input_path}.")
    g = graphs[0]
    if not g.is_homogeneous:
        raise ValueError("Expected a homogeneous DGL graph.")

    x = g.ndata["feature"].float()
    y = g.ndata["label"].long().view(-1)
    src, dst = g.edges()
    edge_index = torch.stack([src, dst], dim=0).long()

    graph = GraphData(x=x, edge_index=edge_index, y=y, name=name)
    graph.aux["source_format"] = "dgl"
    graph.aux["num_edges"] = int(edge_index.size(1))
    graph.aux["normal_count"] = int((y == 0).sum().item())
    graph.aux["anomaly_count"] = int((y != 0).sum().item())

    save_processed_graph(graph, output_path)
    return graph


def main() -> None:
    args = parse_args()
    graph = convert_reddit(args.input, args.output, name=args.name)
    print(f"Saved {args.output}")
    print(
        f"nodes={graph.num_nodes} edges={graph.num_edges} "
        f"features={graph.num_features} anomalies={int((graph.y != 0).sum().item())}"
    )


if __name__ == "__main__":
    main()
