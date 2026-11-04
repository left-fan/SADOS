"""Checks for the Bernstein backbone and PA graph pruning."""

import unittest

import torch

from sados.data import GraphData
from sados.models import build_detector
from sados.models.bwgnn import bernstein_coefficients
from sados.models.ghrn import graph_signature, post_aggregation, prune_edges_by_pa


class GHRNTests(unittest.TestCase):
    def test_bernstein_coefficients_and_trainable_embeddings(self):
        expected = torch.tensor([
            [3.0, -3.0, 0.75],
            [0.0, 3.0, -1.5],
            [0.0, 0.0, 0.75],
        ])
        torch.testing.assert_close(bernstein_coefficients(2), expected)
        x = torch.randn(4, 3)
        edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        model = build_detector("ghrn", in_channels=3, hidden_channels=5, dropout=0.0)
        logits, embedding = model(x, edges, return_embedding=True)
        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertEqual(tuple(embedding.shape), (4, 5))
        logits.square().sum().backward()
        self.assertIsNotNone(model.linear1.weight.grad)

    def test_pa_pruning_keeps_reciprocal_edges_together(self):
        edges = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ])
        probabilities = torch.tensor([
            [0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9],
        ])
        pa = post_aggregation(edges, probabilities)
        self.assertEqual(tuple(pa.shape), (4, 2))
        pruned, stats = prune_edges_by_pa(
            edges, probabilities, delete_ratio=0.25, preserve_undirected=True,
        )
        self.assertEqual(stats["edges_before"], 8)
        self.assertEqual(stats["deleted"], 2)
        kept = set(map(tuple, pruned.t().tolist()))
        for u, v in kept:
            self.assertIn((v, u), kept)

    def test_graph_signature_excludes_labels_but_checks_split(self):
        x = torch.ones(3, 2)
        edges = torch.tensor([[0, 1], [1, 2]])
        masks = (torch.tensor([True, False, False]),
                 torch.tensor([False, True, False]),
                 torch.tensor([False, False, True]))
        a = GraphData(x, edges, torch.tensor([0, 1, 0]),
                      train_mask=masks[0], val_mask=masks[1], test_mask=masks[2])
        b = GraphData(x, edges, torch.tensor([1, 0, 1]),
                      train_mask=masks[0].clone(), val_mask=masks[1].clone(),
                      test_mask=masks[2].clone())
        self.assertEqual(graph_signature(a), graph_signature(b))
        b.x = b.x.clone()
        b.x[0, 0] = 2
        self.assertNotEqual(graph_signature(a), graph_signature(b))
        b.x = x
        b.test_mask = torch.tensor([False, True, True])
        self.assertNotEqual(graph_signature(a), graph_signature(b))


if __name__ == "__main__":
    unittest.main()
