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
from ranpac_critique import (
    BACKBONE_SPECS, CIFAR100_FINE_CLASSES, FLORA_TASK1_NAMES, ClassBalancedRidge,
    ScheduleTaskImages, build_first_task_regimes, build_imbalance_counts,
    compute_embedding_geometry, evaluate_first_vs_later, evaluate_imbalance_diagnostics,
    repartition_features, run_phase2_on_shards, semantic_shift_class_order,
    subsample_task_shard,
)
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

    def test_first_task_bias_regimes_and_repartition(self):
        self.assertEqual(len(CIFAR100_FINE_CLASSES), 100)
        self.assertEqual(list(CIFAR100_FINE_CLASSES), sorted(CIFAR100_FINE_CLASSES))
        flora_order = semantic_shift_class_order(FLORA_TASK1_NAMES, seed=1993)
        self.assertEqual(len(flora_order), 100)
        self.assertEqual(sorted(flora_order), list(range(100)))
        flora_idx_set = {CIFAR100_FINE_CLASSES.index(n) for n in FLORA_TASK1_NAMES}
        self.assertEqual(set(flora_order[:10]), flora_idx_set)

        regimes = build_first_task_regimes(1993)
        for key in ("b10-random", "b5-narrow", "b2-narrow", "b10-flora"):
            self.assertIn(key, regimes)
            self.assertEqual(sum(regimes[key]["task_sizes"]), 100)
            self.assertEqual(sorted(regimes[key]["class_order"]), list(range(100)))

        dummy = SimpleNamespace(targets=[c for c in range(10) for _ in range(4)])
        order10 = [9, 7, 5, 3, 1, 8, 6, 4, 2, 0]
        sizes10 = (2, 4, 4)
        s1 = ScheduleTaskImages(dummy, order10, task=1, task_sizes=sizes10)
        s2 = ScheduleTaskImages(dummy, order10, task=2, task_sizes=sizes10)
        self.assertEqual(sorted(set(s1.labels.tolist())), [0, 1])
        self.assertEqual(sorted(set(s2.labels.tolist())), [2, 3, 4, 5])

        feats = torch.randn(40, 8, generator=torch.Generator().manual_seed(11))
        orig_y = torch.tensor(dummy.targets)
        shards = repartition_features(feats, orig_y, order10, sizes10)
        self.assertEqual([len(x) for x, _ in shards], [8, 16, 16])
        records, _ = run_phase2_on_shards(
            shards, shards, sizes10, method="ridge-rp", projection_dim=24, fixed_lambda=10.0,
        )
        diag = evaluate_first_vs_later(records[-1]["accuracies_pct"], sizes10)
        self.assertEqual(diag["first_task_classes"], 2)
        self.assertEqual(diag["num_tasks"], 3)
        self.assertAlmostEqual(
            diag["overall_100_pct"],
            (2 * records[-1]["accuracies_pct"][0] + 4 * records[-1]["accuracies_pct"][1]
             + 4 * records[-1]["accuracies_pct"][2]) / 10.0,
        )

    def test_class_imbalance_and_balanced_ridge(self):
        step_counts = build_imbalance_counts(10, mode="step", num_classes=20, max_samples=50, classes_per_task=5)
        self.assertEqual(step_counts[:10], [50] * 10)
        self.assertEqual(step_counts[10:], [5] * 10)

        lt_counts = build_imbalance_counts(10, mode="long_tail", num_classes=20, max_samples=50, classes_per_task=5)
        self.assertEqual(max(lt_counts), 50)
        self.assertEqual(min(lt_counts), 5)

        # Equal counts -> ClassBalancedRidge matches standard Ridge identically.
        x = torch.randn(40, 6, generator=torch.Generator().manual_seed(9))
        y = torch.arange(4).repeat_interleave(10)
        std_r = Ridge(6, num_classes=4, projection_dim=16, seed=5)
        cb_r = ClassBalancedRidge(6, num_classes=4, projection_dim=16, seed=5, ref_count=10.0)
        for start in (0, 20):
            std_r.update(std_r.project(x[start:start + 20]), y[start:start + 20])
            cb_r.update(cb_r.project(x[start:start + 20]), y[start:start + 20])
        torch.testing.assert_close(cb_r.G, std_r.G)
        torch.testing.assert_close(cb_r.C, std_r.C)
        torch.testing.assert_close(cb_r.weights(4, 5.0), std_r.weights(4, 5.0))

        # Imbalanced synthetic data: unweighted Ridge shrinks minority weight norms vs CB-Ridge.
        gen = torch.Generator().manual_seed(123)
        centers = F.normalize(torch.randn(4, 8, generator=gen), dim=1) * 2.0
        imb_counts = [40, 40, 2, 2]
        xs, ys = [], []
        for cls_id, cnt in enumerate(imb_counts):
            xs.append(centers[cls_id] + 0.6 * torch.randn(cnt, 8, generator=gen))
            ys.append(torch.full((cnt,), cls_id, dtype=torch.long))
        train_x, train_y = torch.cat(xs), torch.cat(ys)
        sub_x, sub_y = subsample_task_shard(train_x, train_y, [20, 20, 2, 2], seed=1993)
        self.assertEqual(torch.bincount(sub_y, minlength=4).tolist(), [20, 20, 2, 2])

        r_unw = Ridge(8, num_classes=4, projection_dim=32, seed=19)
        r_cb = ClassBalancedRidge(8, num_classes=4, projection_dim=32, seed=19, ref_count=40.0)
        r_unw.update(r_unw.project(train_x), train_y)
        r_cb.update(r_cb.project(train_x), train_y)
        test_shards = [
            (centers[c].repeat(20, 1) + 0.5 * torch.randn(20, 8, generator=gen),
             torch.full((20,), c, dtype=torch.long))
            for c in range(4)
        ]
        d_unw = evaluate_imbalance_diagnostics(
            r_unw, r_unw.weights(4, 10.0), test_shards, imb_counts, torch.device("cpu"),
        )
        d_cb = evaluate_imbalance_diagnostics(
            r_cb, r_cb.weights(4, 10.0), test_shards, imb_counts, torch.device("cpu"),
        )
        self.assertGreater(d_unw["weight_norm_ratio_head_to_tail"], d_cb["weight_norm_ratio_head_to_tail"])

    def test_backbone_specs_and_geometry_diagnostics(self):
        self.assertEqual(
            set(BACKBONE_SPECS),
            {"vit_b16_in21k", "vit_b16_in1k", "vit_b16_dino", "vit_t16_in21k"},
        )
        gen = torch.Generator().manual_seed(2026)
        protos = torch.eye(4, 8) * 3.0
        clean_shards = [
            (protos[c].repeat(15, 1) + 0.1 * torch.randn(15, 8, generator=gen),
             torch.full((15,), c, dtype=torch.long))
            for c in range(4)
        ]
        noisy_shards = [
            (protos[c].repeat(15, 1) + 2.0 * torch.randn(15, 8, generator=gen),
             torch.full((15,), c, dtype=torch.long))
            for c in range(4)
        ]
        g_clean = compute_embedding_geometry(clean_shards, num_classes=4)
        g_noisy = compute_embedding_geometry(noisy_shards, num_classes=4)
        self.assertGreater(g_clean["fisher_trace_ratio"], g_noisy["fisher_trace_ratio"])
        self.assertGreater(g_clean["angular_margin"], g_noisy["angular_margin"])


if __name__ == "__main__":
    unittest.main()
