"""Small checks for the paper's adaptation equations and data boundaries."""

import unittest
from unittest.mock import patch
import sys
import types
from contextlib import redirect_stdout
import io

import torch

from sados.augmentation import build_supervised_tuning_set, shifted_augmentation
from sados.contrastive import ContrastiveConfig, info_nce_loss
from sados.data import GraphData
from sados.models.gcn import GCNDetector
from sados.partition import PartitionConfig, partition_by_confidence
from sados.prototype import PrototypeConfig, update_prototypes
from sados.scheduler import dynamic_supervised_weight
from sados.split import SplitConfig, density_based_split
from sados.trainer_tta import SadosTTATrainer, TTAConfig


class PaperMethodTests(unittest.TestCase):
    def test_class_conditional_top_m_and_supervised_samples(self):
        h = torch.tensor([[1., 0.], [2., 0.], [0., 3.], [0., 4.], [1., 1.]])
        confidence = torch.tensor([.5, .8, .6, .9, .95])
        labels = torch.tensor([0, 0, 1, 1, 0])
        result = update_prototypes(
            h, [torch.tensor([0, 1, 2, 3])], confidence, labels,
            torch.eye(2), config=PrototypeConfig(top_m=1, ema_beta=1.0),
        )
        self.assertEqual(result.state.prototypes.shape, (2, 1, 2))
        torch.testing.assert_close(result.state.prototypes[0, 0], h[1])
        torch.testing.assert_close(result.state.prototypes[1, 0], h[3])
        reliable = torch.tensor([3, 4])
        aug = shifted_augmentation(
            h, reliable, labels, result.state,
            result.group_reliability_bias,
        )
        self.assertEqual(aug.indices.tolist(), [4])
        features, targets, augmented = build_supervised_tuning_set(
            h, reliable, labels, aug,
        )
        self.assertEqual(targets.tolist(), [1, 0, 0])
        self.assertEqual(augmented.tolist(), [False, False, True])
        torch.testing.assert_close(features[:2], h[reliable])

    def test_schedule_and_one_way_contrastive(self):
        self.assertAlmostEqual(dynamic_supervised_weight(0, 10, .4), .4)
        self.assertAlmostEqual(dynamic_supervised_weight(1, 10, .4), .5)
        self.assertAlmostEqual(dynamic_supervised_weight(6, 10, .4), 1.)
        self.assertFalse(ContrastiveConfig().symmetric)
        z = torch.eye(3)
        self.assertLess(info_nce_loss(z, z).item(), 0.1)

    def test_target_labels_do_not_affect_adaptation_step(self):
        torch.manual_seed(7)
        x = torch.randn(12, 3)
        edges = torch.stack([torch.arange(11), torch.arange(1, 12)])
        mask = torch.ones(12, dtype=torch.bool)
        cfg = TTAConfig(
            epochs=2, report_metrics_after_fit=False,
            partition=PartitionConfig.from_dict({
                "thresholds": [.7, .5, .3, .1, 0.],
                "calibration": {"lambda_mi": 0.},
            }),
            contrastive=ContrastiveConfig(max_nodes=8, negative_sample_size=3),
        )
        model1 = GCNDetector(3, hidden_channels=4, dropout=0.)
        model2 = GCNDetector(3, hidden_channels=4, dropout=0.)
        model2.load_state_dict(model1.state_dict())
        graph1 = GraphData(x.clone(), edges.clone(), torch.zeros(12, dtype=torch.long), test_mask=mask)
        graph2 = GraphData(x.clone(), edges.clone(), torch.ones(12, dtype=torch.long), test_mask=mask)
        trainer1 = SadosTTATrainer(model1, graph1, cfg)
        trainer2 = SadosTTATrainer(model2, graph2, cfg)
        torch.manual_seed(11)
        trainer1.adapt_step(target_idx=torch.arange(12), step=1, total_steps=2)
        torch.manual_seed(11)
        trainer2.adapt_step(target_idx=torch.arange(12), step=1, total_steps=2)
        for a, b in zip(model1.parameters(), model2.parameters()):
            torch.testing.assert_close(a, b)
        with self.assertRaises(ValueError):
            SadosTTATrainer(
                GCNDetector(3), graph1,
                TTAConfig(monitor="target_auroc"),
            )

    def test_density_split_uses_metis(self):
        graph = GraphData(
            torch.ones(6, 2),
            torch.tensor([[0, 0, 0, 0, 1, 1, 2, 4],
                          [1, 2, 3, 4, 2, 3, 3, 5]]),
            torch.tensor([0, 0, 0, 1, 0, 1]),
        )
        calls = []
        fake = types.SimpleNamespace(
            Options=lambda seed: types.SimpleNamespace(seed=seed),
            part_graph=lambda k, adjacency, options: (
                calls.append((k, adjacency, options.seed)) or 2,
                [0, 0, 0, 0, 1, 1],
            ),
        )
        with patch.dict(sys.modules, {"pymetis": fake}):
            result = density_based_split(graph, SplitConfig(target_density="high"))
        self.assertEqual(calls[0][0], 2)
        self.assertEqual(calls[0][2], 42)
        self.assertEqual(result.meta["partitioner"], "METIS")
        self.assertEqual(result.target_idx.tolist(), [0, 1, 2, 3])

    def test_each_tta_round_logs_test_metrics(self):
        torch.manual_seed(17)
        graph = GraphData(
            torch.randn(8, 3),
            torch.stack([torch.arange(7), torch.arange(1, 8)]),
            torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
            test_mask=torch.ones(8, dtype=torch.bool),
        )
        cfg = TTAConfig(
            epochs=2, report_metrics_after_fit=False,
            partition=PartitionConfig.from_dict({"calibration": {"lambda_mi": 0.}}),
            contrastive=ContrastiveConfig(max_nodes=8, negative_sample_size=3),
        )
        output = io.StringIO()
        with redirect_stdout(output):
            history = SadosTTATrainer(
                GCNDetector(3, hidden_channels=4, dropout=0.), graph, cfg,
            ).fit()["history"]
        lines = output.getvalue().strip().splitlines()
        self.assertEqual(len(lines), 2)
        for row, line in zip(history, lines):
            for key in ("target_auroc", "target_auprc", "target_f1"):
                self.assertIn(key, row)
                self.assertIn(key + "=", line)


if __name__ == "__main__":
    unittest.main()
