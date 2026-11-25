"""CPU checks for the Phase 2 protocol, ridge sweep and checkpoint state."""

import io
import unittest
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from ranpac_core import (
    ACTIVATIONS, NCM, Ridge, apply_activation, evaluate_seen, gram_spectrum_stats,
    select_ridge, split_class_order,
)
from ranpac_colab import LINEAR_DIMS, PRESETS, TABLE_A5_PAPER, method_activation
from ranpac_data import TaskImages
from ranpac_full import PHASE1_ABLATION_PRESETS


class Phase2Tests(unittest.TestCase):
    def test_class_order_and_remapping(self):
        order = split_class_order()
        self.assertEqual(order[:10], [68, 56, 78, 8, 23, 84, 90, 65, 74, 76])
        self.assertEqual(sorted(order), list(range(100)))
        original = SimpleNamespace(targets=list(range(100)))
        shards = [TaskImages(original, order, task) for task in range(1, 11)]
        for task, shard in enumerate(shards):
            self.assertEqual(sorted(shard.labels.tolist()), list(range(10 * task, 10 * task + 10)))
            self.assertEqual(set(shard.indices.tolist()), set(order[10 * task:10 * task + 10]))

    def test_ncm_uses_all_seen_classes_for_each_test_shard(self):
        ncm = NCM(2, num_classes=2)
        ncm.update(torch.tensor([[2., 0.], [4., 0.]]), torch.tensor([0, 0]))
        self.assertEqual(ncm.weights(1).shape, (2, 1))
        ncm.update(torch.tensor([[0., 3.]]), torch.tensor([1]))
        torch.testing.assert_close(ncm.weights(2), torch.eye(2))
        scores = evaluate_seen(ncm, ncm.weights(2), [
            (torch.tensor([[0., 1.]]), torch.tensor([0])),
            (torch.tensor([[0., 1.]]), torch.tensor([1])),
        ], torch.device("cpu"), batch_size=1)
        self.assertEqual(scores, [0.0, 100.0])

    def test_ridge_accumulation_and_resume_match_joint_solve(self):
        features = torch.randn(16, 5, generator=torch.Generator().manual_seed(77))
        labels = torch.tensor([0, 1] * 4 + [2, 3] * 4)
        ridge = 0.7
        for projection_dim in (0, 11):
            with self.subTest(projection_dim=projection_dim):
                model = Ridge(5, num_classes=4, projection_dim=projection_dim, seed=3)
                projected = features if model.W is None else (features @ model.W).relu()
                torch.testing.assert_close(model.project(features), projected)
                model.update(model.project(features[:8]), labels[:8])
                self.assertEqual(model.logits(features, model.weights(2, ridge)).shape, (16, 2))

                checkpoint = io.BytesIO()
                torch.save(model.state_dict(), checkpoint)
                checkpoint.seek(0)
                resumed = Ridge(5, num_classes=4, projection_dim=projection_dim, seed=999)
                resumed.load_state_dict(torch.load(checkpoint, weights_only=True))
                for classifier in (model, resumed):
                    for start in (8, 12):
                        classifier.update(classifier.project(features[start:start + 4]), labels[start:start + 4])

                gram = projected.T @ projected
                cross = projected.T @ F.one_hot(labels, num_classes=4).float()
                expected = projected @ torch.linalg.solve(gram + ridge * torch.eye(len(gram)), cross)
                torch.testing.assert_close(resumed.G, gram)
                torch.testing.assert_close(resumed.C, cross)
                torch.testing.assert_close(resumed.logits(features, resumed.weights(4, ridge)), expected)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(resumed.state_dict()[key], value, atol=0, rtol=0)

    def test_primal_and_dual_lambda_selection_match_direct_solution(self):
        seed, task = 1993, 2
        grid = (0.01, 0.1, 1.0, 10.0, 100.0)
        order = torch.randperm(20, generator=torch.Generator().manual_seed(seed + task))
        labels = torch.arange(20) % 3
        target = F.one_hot(labels[order], num_classes=3).double()
        for dim in (6, 26):
            with self.subTest(dim=dim):
                features = torch.randn(20, dim, generator=torch.Generator().manual_seed(15))
                train, val = features[order].double().split([16, 4])
                gram, cross = train.T @ train, train.T @ target[:16]
                scores = []
                for ridge in grid:
                    weight = torch.linalg.solve(gram + ridge * torch.eye(dim, dtype=gram.dtype), cross)
                    scores.append(F.mse_loss(val @ weight, target[16:]).item())
                self.assertEqual(select_ridge(features, labels, 3, seed, task, grid), grid[scores.index(min(scores))])

    def test_activations_presets_and_gram_spectrum(self):
        self.assertIn(768, LINEAR_DIMS)
        self.assertIn("rp-linear-m768", PRESETS)
        for dim, acc in TABLE_A5_PAPER.items():
            self.assertEqual(PRESETS[f"rp-a5-m{dim}"], ("ridge-rp", dim, 100.0, acc))
        self.assertEqual(set(PHASE1_ABLATION_PRESETS), {"phase1-ncm", "phase1-no-rp"})

        features = torch.randn(64, 12, generator=torch.Generator().manual_seed(42))
        for act in ACTIVATIONS:
            with self.subTest(activation=act):
                model = Ridge(12, num_classes=4, projection_dim=32, seed=7, activation=act)
                raw = features @ model.W
                expected = apply_activation(raw.clone(), act, inplace=False)
                actual = model.project(features.clone())
                torch.testing.assert_close(actual, expected)
                method = "ridge-rp" if act == "relu" else f"ridge-rp-{act}"
                self.assertEqual(method_activation(method), act)

        lin = Ridge(12, num_classes=4, projection_dim=48, seed=7, use_relu=False)
        lin.update(lin.project(features), torch.arange(64) % 4)
        lin_stats = gram_spectrum_stats(lin.G)
        self.assertLessEqual(lin_stats["rank_standard"], 12)
        self.assertLessEqual(lin_stats["rank_deflated_dc"], 12)

        rel = Ridge(12, num_classes=4, projection_dim=48, seed=7, activation="relu")
        rel.update(rel.project(features), torch.arange(64) % 4)
        rel_stats = gram_spectrum_stats(rel.G)
        self.assertGreater(rel_stats["rank_deflated_dc"], 12)


if __name__ == "__main__":
    unittest.main()
