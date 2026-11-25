"""Critique & stress-test runners for RanPAC (Notebooks 11, 12, 13).

Covers three empirical weakness studies on Split CIFAR-100 without modifying the
core reproduction modules or their feature-cache SHA256 digests:
  A. First-Task Bias in Phase 1 (narrow B in {2, 5} & semantic-shifted Task 1).
  B. Class Imbalance sensitivity in Phase 2 (unweighted Ridge+RP vs NCM & CB-Ridge).
  C. Pre-trained Backbone dependency (IN-21K vs IN-1K vs DINO vs ViT-Tiny).
"""

import csv
import hashlib
import json
from contextlib import nullcontext
from importlib.metadata import version
from itertools import chain
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ranpac_colab import configure_wandb, export_results, save_checkpoint
from ranpac_core import (
    CLASSES_PER_TASK, NUM_CLASSES, NUM_TASKS, RIDGE_GRID, SEED, NCM, Ridge,
    apply_activation, evaluate_seen, select_ridge, split_class_order,
)
from ranpac_data import (
    CHECKPOINT_URL, FEATURE_DIM, MODEL_ID, FeatureCache, atomic_output, feature_config,
)
from ranpac_full import (
    BATCH_SIZE, LR, MOMENTUM, WEIGHT_DECAY, CosineHead, adapter_sha256,
    build_adapted_backbone, train_epoch,
)

try:
    import resource
except ImportError:
    resource = None


CIFAR100_FINE_CLASSES = (
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
)

FLORA_TASK1_NAMES = (
    "maple_tree", "oak_tree", "orchid", "palm_tree", "pine_tree",
    "poppy", "rose", "sunflower", "tulip", "willow_tree",
)
AQUATIC_TASK1_NAMES = (
    "aquarium_fish", "beaver", "dolphin", "flatfish", "otter",
    "ray", "seal", "shark", "trout", "whale",
)


def _validate_run_label(run_label):
    if not run_label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run_label):
        raise ValueError("Use lowercase letters, digits, '-' or '_' in run_label.")


def _host_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 if resource else None


# ============================================================================
# Plan A: First-Task Bias in Phase 1 (Task 1 size & semantic shift)
# ============================================================================


def semantic_shift_class_order(task1_names=FLORA_TASK1_NAMES, seed=SEED,
                               class_names=CIFAR100_FINE_CLASSES):
    """Place specific fine classes in Task 1 and permute all remaining classes."""
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    missing = [name for name in task1_names if name not in name_to_idx]
    if missing:
        raise ValueError(f"Unknown CIFAR-100 fine class names: {missing}")
    rng = np.random.RandomState(seed)
    first_indices = [name_to_idx[name] for name in task1_names]
    first_indices = [first_indices[i] for i in rng.permutation(len(first_indices))]
    first_set = set(first_indices)
    rest_pool = [idx for idx in range(len(class_names)) if idx not in first_set]
    rest_indices = [rest_pool[i] for i in rng.permutation(len(rest_pool))]
    order = first_indices + rest_indices
    if len(order) != len(class_names) or len(set(order)) != len(class_names):
        raise ValueError("Invalid semantic-shift class permutation.")
    return order


def build_first_task_regimes(seed=SEED):
    """Return canonical First-Task Bias regimes covering all 100 CIFAR-100 classes."""
    std_order = split_class_order(seed)
    return {
        "b10-random": {
            "title": "B=10 ngẫu nhiên (Chuẩn Paper)",
            "class_order": std_order,
            "task_sizes": (10,) * 10,
            "description": "Task 1 gồm 10 lớp ngẫu nhiên đa dạng, theo sau bởi 9 task × 10 lớp.",
        },
        "b5-narrow": {
            "title": "B=5 hẹp (20 task × 5 lớp)",
            "class_order": std_order,
            "task_sizes": (5,) * 20,
            "description": "Task 1 chỉ gồm 5 lớp đầu (2.500 ảnh), theo sau bởi 19 task × 5 lớp.",
        },
        "b2-narrow": {
            "title": "B=2 rất hẹp (Task 1 = 2 lớp, 7 task sau × 14 lớp)",
            "class_order": std_order,
            "task_sizes": (2,) + (14,) * 7,
            "description": "Task 1 chỉ có 2 lớp (1.000 ảnh), 98 lớp còn lại chia đều vào 7 task.",
        },
        "b10-flora": {
            "title": "B=10 lệch ngữ nghĩa (Task 1 chỉ Hoa + Cây gỗ)",
            "class_order": semantic_shift_class_order(FLORA_TASK1_NAMES, seed=seed),
            "task_sizes": (10,) * 10,
            "description": "Task 1 đủ 10 lớp nhưng đồng nhất miền thực vật; Tasks 2..10 là 90 lớp khác miền.",
        },
    }


def task_boundaries(task_sizes):
    """Return cumulative (start, stop) class intervals for a 1-indexed task schedule."""
    sizes = tuple(int(s) for s in task_sizes)
    if not sizes or any(s < 1 for s in sizes):
        raise ValueError("All task_sizes entries must be positive integers.")
    bounds = []
    cursor = 0
    for size in sizes:
        bounds.append((cursor, cursor + size))
        cursor += size
    return tuple(bounds)


class ScheduleTaskImages(Dataset):
    """Dataset view for task `task` (1-indexed) under arbitrary `task_sizes`."""

    def __init__(self, dataset, class_order, task, task_sizes):
        bounds = task_boundaries(task_sizes)
        num_classes = len(class_order)
        if bounds[-1][1] != num_classes:
            raise ValueError("sum(task_sizes) must equal len(class_order).")
        if not 1 <= task <= len(bounds):
            raise ValueError(f"task must be in 1..{len(bounds)}, got {task}.")
        old_to_new = np.empty(num_classes, dtype=np.int64)
        old_to_new[class_order] = np.arange(num_classes)
        labels = old_to_new[np.asarray(dataset.targets)]
        start, stop = bounds[task - 1]
        self.indices = np.flatnonzero((labels >= start) & (labels < stop))
        self.labels = labels[self.indices]
        self.dataset = dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.dataset[int(self.indices[index])]
        return image, int(self.labels[index])


def load_full_cached_cifar100(cache):
    """Load all 10 standard CIFAR-100 shards and restore original CIFAR-100 labels 0..99."""
    base_order = np.asarray(cache.config["class_order"], dtype=np.int64)
    out = {}
    for split in ("train", "test"):
        feat_list, orig_label_list = [], []
        for task in range(1, NUM_TASKS + 1):
            feats, remapped_labels, _ = cache.load(split, task)
            feat_list.append(feats)
            orig_label_list.append(torch.from_numpy(base_order[remapped_labels.numpy()]))
        out[split] = (torch.cat(feat_list, dim=0), torch.cat(orig_label_list, dim=0))
    return out


def repartition_features(features, orig_labels, class_order, task_sizes):
    """Remap original labels 0..K-1 to `class_order` and slice into `len(task_sizes)` shards."""
    bounds = task_boundaries(task_sizes)
    num_classes = len(class_order)
    if bounds[-1][1] != num_classes:
        raise ValueError("sum(task_sizes) must equal len(class_order).")
    old_to_new = torch.empty(num_classes, dtype=torch.long)
    old_to_new[torch.as_tensor(class_order, dtype=torch.long)] = torch.arange(num_classes)
    new_labels = old_to_new[orig_labels]
    shards = []
    for start, stop in bounds:
        mask = (new_labels >= start) & (new_labels < stop)
        shards.append((features[mask], new_labels[mask]))
    return shards


def evaluate_first_vs_later(accuracies_pct, task_sizes):
    """Compute exact class-weighted overall, Task-1, and Tasks-2..T accuracies on balanced test classes."""
    sizes = [int(s) for s in task_sizes]
    if len(accuracies_pct) != len(sizes):
        raise ValueError("Need accuracy for every task in task_sizes.")
    total_classes = sum(sizes)
    overall_pct = sum(acc * sz for acc, sz in zip(accuracies_pct, sizes)) / total_classes
    task1_pct = float(accuracies_pct[0])
    later_classes = total_classes - sizes[0]
    later_pct = (
        sum(acc * sz for acc, sz in zip(accuracies_pct[1:], sizes[1:])) / later_classes
        if later_classes > 0 else float("nan")
    )
    return {
        "first_task_classes": sizes[0],
        "num_tasks": len(sizes),
        "avg_task_acc_pct": float(sum(accuracies_pct) / len(accuracies_pct)),
        "overall_100_pct": float(overall_pct),
        "task1_acc_pct": task1_pct,
        "later_classes_acc_pct": float(later_pct),
        "task1_minus_later_gap_pp": float(task1_pct - later_pct),
    }


def run_phase2_on_shards(train_shards, test_shards, task_sizes, *, method="ridge-rp",
                         projection_dim=10000, fixed_lambda=100.0, stats_device="cpu",
                         seed=SEED, activation="relu"):
    """Run Phase 2 over pre-loaded `(features, labels)` shards with variable `task_sizes`."""
    bounds = task_boundaries(task_sizes)
    num_classes = bounds[-1][1]
    feature_dim = train_shards[0][0].shape[1]
    stats_dev = torch.device(stats_device)
    classifier = (
        NCM(feature_dim, num_classes=num_classes) if method == "ncm" else
        Ridge(feature_dim, num_classes=num_classes, projection_dim=0, seed=seed) if method == "ridge" else
        Ridge(feature_dim, num_classes=num_classes, projection_dim=projection_dim,
              seed=seed, activation=activation)
    ).to(stats_dev)

    records = []
    for task_idx, ((train_x, train_y), (_, stop)) in enumerate(zip(train_shards, bounds), start=1):
        started = perf_counter()
        x_dev, y_dev = train_x.to(stats_dev), train_y.to(stats_dev)
        ridge_val = lambda_sec = solve_sec = None
        if isinstance(classifier, Ridge):
            projected = classifier.project(x_dev)
            ridge_val = fixed_lambda
            if ridge_val is None:
                if stats_dev.type == "cuda":
                    torch.cuda.synchronize(stats_dev)
                t0 = perf_counter()
                ridge_val = select_ridge(projected, y_dev, num_classes, seed, task_idx)
                if stats_dev.type == "cuda":
                    torch.cuda.synchronize(stats_dev)
                lambda_sec = perf_counter() - t0
            classifier.update(projected, y_dev)
            del projected
            if stats_dev.type == "cuda":
                torch.cuda.synchronize(stats_dev)
            t0 = perf_counter()
            weights = classifier.weights(stop, ridge_val)
            if stats_dev.type == "cuda":
                torch.cuda.synchronize(stats_dev)
            solve_sec = perf_counter() - t0
        else:
            classifier.update(x_dev, y_dev)
            weights = classifier.weights(stop)
        fit_sec = perf_counter() - started
        del x_dev, y_dev

        eval_t0 = perf_counter()
        accs = evaluate_seen(classifier, weights, test_shards[:task_idx], stats_dev)
        eval_sec = perf_counter() - eval_t0
        seen_sizes = task_sizes[:task_idx]
        class_weighted_acc = sum(a * s for a, s in zip(accs, seen_sizes)) / sum(seen_sizes)
        records.append({
            "task": task_idx,
            "seen_classes": stop,
            "avg_accuracy_pct": sum(accs) / len(accs),
            "class_weighted_accuracy_pct": class_weighted_acc,
            "accuracies_pct": accs,
            "ridge_lambda": ridge_val,
            "lambda_search_seconds": lambda_sec,
            "ridge_solve_seconds": solve_sec,
            "fit_seconds": fit_sec,
            "eval_seconds": eval_sec,
            "task_seconds": perf_counter() - started,
            "host_peak_rss_mib": _host_rss_mib(),
        })
    return records, classifier


def extract_all_cifar100_with_backbone(backbone, data_root, device, *, batch_size=48,
                                       workers=2, eval_transform=None):
    """Extract `(features, orig_labels)` for all 50k train and 10k test CIFAR-100 images."""
    from torchvision import datasets, transforms

    if eval_transform is None:
        eval_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    backbone.eval().requires_grad_(False)
    out = {}
    for split in ("train", "test"):
        ds = datasets.CIFAR100(root=data_root, train=(split == "train"),
                               transform=eval_transform, download=True)
        if tuple(ds.classes) != CIFAR100_FINE_CLASSES:
            raise ValueError("CIFAR-100 fine class list differs from expected canonical order.")
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=workers,
            pin_memory=(device.type == "cuda"),
        )
        feat_batches, label_batches = [], []
        with torch.inference_mode():
            for images, targets in loader:
                emb = backbone(images.to(device, non_blocking=True))
                feat_batches.append(emb.float().cpu())
                label_batches.append(targets.long().cpu())
        feats = torch.cat(feat_batches, dim=0)
        labels = torch.cat(label_batches, dim=0)
        if not torch.isfinite(feats).all():
            raise ValueError(f"Non-finite embeddings encountered in {split} split.")
        out[split] = (feats, labels)
    return out


def run_first_task_regime(regime_key, work_dir, *, regimes=None, run_label="ftb-01",
                          phase1_epochs=20, microbatch=48, precision="fp32",
                          temporary_head="cosine", projection_dim=10000,
                          fixed_lambda=100.0, feature_batch_size=48,
                          stats_device="cpu", workers=2, data_root="/content/ranpac-data",
                          reuse_nb04_adapter_path=None, entity=None,
                          project="RanPAC-reproduction", wandb_mode="online"):
    """Train/load Phase 1 for `regime_key`, evaluate Phase 2, and pair with No-Phase-1 control."""
    _validate_run_label(run_label)
    regimes = regimes or build_first_task_regimes(SEED)
    if regime_key not in regimes:
        raise ValueError(f"Unknown regime {regime_key!r}. Choose from {tuple(regimes)}.")
    spec = regimes[regime_key]
    class_order = list(spec["class_order"])
    task_sizes = tuple(int(s) for s in spec["task_sizes"])
    first_k = task_sizes[0]

    work_dir = Path(work_dir)
    run_name = f"ftb-{regime_key}-s{SEED}-{run_label}"
    run_dir = work_dir / "runs" / run_name
    ckpt_path = run_dir / "checkpoint.pt"

    if ckpt_path.is_file():
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if saved.get("completed"):
            print(f"[Đã xong] {run_name} -> Overall 100-class = "
                  f"{saved['summary']['phase1_overall_100_pct']:.2f}% "
                  f"(No-Phase1: {saved['summary']['frozen_overall_100_pct']:.2f}%)")
            return saved

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab before running First-Task Bias Phase 1.")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")

    # 1. Load or extract frozen baseline embeddings once from the shared standard FeatureCache.
    std_cache = FeatureCache(
        work_dir / "embeddings", data_root,
        feature_config(split_class_order(SEED), device),
        device, feature_batch_size, workers,
    )
    frozen_full = load_full_cached_cifar100(std_cache)
    del std_cache
    frozen_train = repartition_features(*frozen_full["train"], class_order, task_sizes)
    frozen_test = repartition_features(*frozen_full["test"], class_order, task_sizes)
    del frozen_full

    frozen_records, _ = run_phase2_on_shards(
        frozen_train, frozen_test, task_sizes,
        method="ridge-rp", projection_dim=projection_dim,
        fixed_lambda=fixed_lambda, stats_device=stats_device, seed=SEED,
    )
    del frozen_train, frozen_test
    frozen_diag = evaluate_first_vs_later(frozen_records[-1]["accuracies_pct"], task_sizes)

    # 2. Train or reuse Phase 1 AdaptFormer on Task 1 only.
    config = {
        "experiment": "first_task_bias",
        "regime": regime_key,
        "regime_title": spec["title"],
        "class_order": class_order,
        "task_sizes": list(task_sizes),
        "first_task_classes": first_k,
        "seed": SEED,
        "phase1_epochs": phase1_epochs,
        "microbatch": microbatch,
        "precision": precision,
        "temporary_head": temporary_head,
        "M": projection_dim,
        "lambda_mode": "task_cv" if fixed_lambda is None else "fixed",
        "lambda_fixed": fixed_lambda,
        "stats_device": stats_device,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_path.is_file():
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    else:
        checkpoint = {
            "config": config,
            "wandb_id": uuid4().hex,
            "wandb_initialized": False,
            "adapter": None,
            "phase1_train_state": None,
            "epochs": [],
            "phase1_records": [],
            "frozen_records": frozen_records,
            "completed": False,
            "summary": None,
        }
        save_checkpoint(ckpt_path, checkpoint)

    # Optional reuse of Notebook 04's trained adapter when regime is standard b10-random.
    reused_nb04 = False
    if (
        regime_key == "b10-random"
        and reuse_nb04_adapter_path is not None
        and Path(reuse_nb04_adapter_path).is_file()
        and len(checkpoint["epochs"]) < phase1_epochs
    ):
        nb04 = torch.load(reuse_nb04_adapter_path, map_location="cpu", weights_only=True)
        if nb04.get("adapter") is not None and len(nb04.get("epochs", ())) == phase1_epochs:
            checkpoint["adapter"] = nb04["adapter"]
            checkpoint["epochs"] = nb04["epochs"]
            checkpoint["phase1_train_state"] = None
            reused_nb04 = True
            save_checkpoint(ckpt_path, checkpoint)
            print(f"[Tái sử dụng Phase 1] Nạp adapter 20 epoch từ {reuse_nb04_adapter_path}")

    wandb = None if wandb_mode == "disabled" else configure_wandb(entity, wandb_mode)
    run_ctx = nullcontext()
    if wandb is not None:
        resume_opt = "must" if checkpoint["wandb_initialized"] else "allow"
        run_ctx = wandb.init(
            entity=entity, project=project, id=checkpoint["wandb_id"], name=run_name,
            group=f"cifar100-first-task-bias-{run_label}",
            tags=["empirics", "critique", "first-task-bias", regime_key],
            config=config, mode=wandb_mode, resume=resume_opt,
            dir=str(run_dir) if wandb_mode == "offline" else "/content",
        )

    with run_ctx as run:
        if run is not None:
            checkpoint["wandb_initialized"] = True
            checkpoint["wandb_url"] = run.url
            run.define_metric("phase1/*", step_metric="phase1/epoch")
            run.define_metric("phase2/*", step_metric="phase2/task")
            save_checkpoint(ckpt_path, checkpoint)

        torch.manual_seed(SEED)
        backbone = build_adapted_backbone(device)
        adapters = nn.ModuleList(block.adapter for block in backbone.blocks)
        if checkpoint["adapter"] is not None:
            adapters.load_state_dict(checkpoint["adapter"])

        if len(checkpoint["epochs"]) < phase1_epochs:
            from torchvision import datasets, transforms

            train_tf = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3 / 4, 4 / 3)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            cifar_train = datasets.CIFAR100(data_root, train=True, transform=train_tf, download=True)
            task1_ds = ScheduleTaskImages(cifar_train, class_order, task=1, task_sizes=task_sizes)

            head = (CosineHead(first_k) if temporary_head == "cosine"
                    else nn.Linear(FEATURE_DIM, first_k)).to(device)
            optimizer = torch.optim.SGD(
                chain(adapters.parameters(), head.parameters()),
                lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_epochs)
            scaler = torch.amp.GradScaler(device.type, enabled=(precision == "amp-fp16"))
            if checkpoint["phase1_train_state"] is not None:
                st = checkpoint["phase1_train_state"]
                head.load_state_dict(st["head"])
                optimizer.load_state_dict(st["optimizer"])
                scheduler.load_state_dict(st["scheduler"])
                scaler.load_state_dict(st["scaler"])

            for epoch in range(len(checkpoint["epochs"]) + 1, phase1_epochs + 1):
                rec = train_epoch(
                    backbone, head, task1_ds, optimizer, scaler, epoch,
                    device=device, microbatch=microbatch, workers=workers,
                )
                scheduler.step()
                checkpoint["adapter"] = adapters.state_dict()
                checkpoint["phase1_train_state"] = (
                    {
                        "head": head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                    } if epoch < phase1_epochs else None
                )
                checkpoint["epochs"].append(rec)
                save_checkpoint(ckpt_path, checkpoint)
                if run is not None:
                    run.log({f"phase1/{k}": v for k, v in rec.items() if v is not None})
                print(f"[{regime_key}] Epoch {epoch:02}/{phase1_epochs} | "
                      f"loss={rec['train_loss']:.4f} | acc={rec['train_accuracy_pct']:.2f}% | "
                      f"{rec['epoch_seconds']:.1f}s")
            del head, optimizer, scheduler, task1_ds, cifar_train

        # 3. Extract or reuse adapted CIFAR-100 embeddings (cached by adapter SHA256).
        adapted_key = hashlib.sha256(json.dumps({
            "dataset": "CIFAR-100-full-orig",
            "model_id": MODEL_ID,
            "checkpoint": CHECKPOINT_URL,
            "adapter_sha256": adapter_sha256(checkpoint["adapter"]),
        }, sort_keys=True).encode()).hexdigest()[:16]
        adapted_cache_file = work_dir / "embeddings" / f"adapted-full-{adapted_key}.pt"

        # Also check if Notebook 04's standard 10-task adapted cache already exists!
        if reused_nb04 and reuse_nb04_adapter_path is not None and not adapted_cache_file.is_file():
            nb04_cfg = torch.load(reuse_nb04_adapter_path, map_location="cpu", weights_only=True)["config"]
            nb04_cache = FeatureCache(
                work_dir / "embeddings", data_root,
                {
                    **feature_config(split_class_order(SEED), torch.device("cuda")),
                    "adapter_sha256": adapter_sha256(checkpoint["adapter"]),
                    "implementation_sha256": nb04_cfg["source_sha256"],
                },
                device, feature_batch_size, workers,
            )
            if all((nb04_cache.root / f"{sp}-task-{t:02}.pt").is_file()
                   for t in range(1, 11) for sp in ("train", "test")):
                adapted_full = load_full_cached_cifar100(nb04_cache)
                with atomic_output(adapted_cache_file) as tmp:
                    torch.save(adapted_full, tmp)
                print(f"[Tái sử dụng Cache] Khớp toàn bộ 20 shard adapted của Notebook 04.")

        if adapted_cache_file.is_file():
            adapted_full = torch.load(adapted_cache_file, map_location="cpu", weights_only=True)
        else:
            print(f"[{regime_key}] Trích xuất đặc trưng đã thích nghi cho 100 lớp CIFAR-100...")
            adapted_full = extract_all_cifar100_with_backbone(
                backbone, data_root, device, batch_size=feature_batch_size, workers=workers,
            )
            with atomic_output(adapted_cache_file) as tmp:
                torch.save(adapted_full, tmp)

        del backbone, adapters
        torch.cuda.empty_cache()

        p1_train = repartition_features(*adapted_full["train"], class_order, task_sizes)
        p1_test = repartition_features(*adapted_full["test"], class_order, task_sizes)
        del adapted_full

        phase1_records, _ = run_phase2_on_shards(
            p1_train, p1_test, task_sizes,
            method="ridge-rp", projection_dim=projection_dim,
            fixed_lambda=fixed_lambda, stats_device=stats_device, seed=SEED,
        )
        del p1_train, p1_test

        p1_diag = evaluate_first_vs_later(phase1_records[-1]["accuracies_pct"], task_sizes)
        summary = {
            "regime": regime_key,
            "title": spec["title"],
            "first_task_classes": first_k,
            "num_tasks": len(task_sizes),
            "frozen_overall_100_pct": frozen_diag["overall_100_pct"],
            "frozen_task1_pct": frozen_diag["task1_acc_pct"],
            "frozen_later_pct": frozen_diag["later_classes_acc_pct"],
            "phase1_overall_100_pct": p1_diag["overall_100_pct"],
            "phase1_avg_task_pct": p1_diag["avg_task_acc_pct"],
            "phase1_task1_pct": p1_diag["task1_acc_pct"],
            "phase1_later_pct": p1_diag["later_classes_acc_pct"],
            "phase1_task1_minus_later_pp": p1_diag["task1_minus_later_gap_pp"],
            "gain_overall_pp": p1_diag["overall_100_pct"] - frozen_diag["overall_100_pct"],
            "gain_task1_pp": p1_diag["task1_acc_pct"] - frozen_diag["task1_acc_pct"],
            "gain_later_pp": p1_diag["later_classes_acc_pct"] - frozen_diag["later_classes_acc_pct"],
            "wandb_url": run.url if run is not None else None,
        }

        if run is not None:
            for rec in phase1_records:
                run.log({
                    "phase2/task": rec["task"],
                    "phase2/seen_classes": rec["seen_classes"],
                    "phase2/avg_accuracy_pct": rec["avg_accuracy_pct"],
                    "phase2/class_weighted_accuracy_pct": rec["class_weighted_accuracy_pct"],
                })
            for k, v in summary.items():
                if isinstance(v, (int, float)):
                    run.summary[k] = v

        checkpoint["phase1_records"] = phase1_records
        checkpoint["frozen_records"] = frozen_records
        checkpoint["summary"] = summary
        checkpoint["completed"] = True
        save_checkpoint(ckpt_path, checkpoint)
        export_results(run_dir, phase1_records)
        with atomic_output(run_dir / "summary.json") as tmp:
            tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print(
            f"[{regime_key}] Xong! Overall 100-class: {summary['phase1_overall_100_pct']:.2f}% "
            f"(Δ vs Frozen: {summary['gain_overall_pp']:+.2f} pp | "
            f"Δ Task 1: {summary['gain_task1_pp']:+.2f} pp | "
            f"Δ Later: {summary['gain_later_pp']:+.2f} pp)"
        )
    return checkpoint


# ============================================================================
# Plan B: Class Imbalance Sensitivity in Phase 2 (Unweighted Ridge vs NCM/CB)
# ============================================================================


def build_imbalance_counts(imbalance_ratio, mode="step", *, num_classes=NUM_CLASSES,
                           max_samples=500, classes_per_task=CLASSES_PER_TASK):
    """Build deterministic per-class training sample counts for 0..num_classes-1.

    Modes:
      - 'step': First half of classes (Tasks 1..5) have `max_samples`, second half
                (Tasks 6..10) have `round(max_samples / imbalance_ratio)`.
      - 'long_tail': Intra-task interleaved exponential profile so every task contains
                     both head and tail classes (`n_r = max_samples * ratio^(-r/(K-1))`).
    """
    if imbalance_ratio < 1.0:
        raise ValueError("imbalance_ratio must be >= 1.0.")
    if mode == "step":
        half = num_classes // 2
        tail_n = max(1, int(round(max_samples / float(imbalance_ratio))))
        return [int(max_samples) if c < half else tail_n for c in range(num_classes)]
    if mode == "long_tail":
        num_tasks = num_classes // classes_per_task
        counts = [0] * num_classes
        for task_zero in range(num_tasks):
            for slot in range(classes_per_task):
                cls_id = task_zero * classes_per_task + slot
                rank = slot * num_tasks + task_zero
                n_c = int(round(max_samples * (float(imbalance_ratio) ** (-rank / max(1, num_classes - 1)))))
                counts[cls_id] = max(1, n_c)
        return counts
    raise ValueError(f"Unsupported imbalance mode {mode!r}. Choose 'step' or 'long_tail'.")


def subsample_task_shard(features, labels, class_counts, *, seed=SEED):
    """Deterministically subsample each class `c` in `(features, labels)` to `class_counts[c]`."""
    selected = []
    unique_classes = torch.unique(labels, sorted=True).tolist()
    for cls_id in unique_classes:
        cls_idx = torch.nonzero(labels == cls_id, as_tuple=False).squeeze(1)
        target_n = int(class_counts[int(cls_id)])
        if target_n > len(cls_idx):
            raise ValueError(f"Requested {target_n} samples for class {cls_id}, only {len(cls_idx)} available.")
        if target_n == len(cls_idx):
            selected.append(cls_idx)
        else:
            gen = torch.Generator().manual_seed(int(seed) * 1009 + int(cls_id))
            perm = torch.randperm(len(cls_idx), generator=gen)
            selected.append(cls_idx[perm[:target_n]])
    keep = torch.cat(selected, dim=0)
    keep, _ = torch.sort(keep)
    return features[keep], labels[keep]


class ClassBalancedRidge(nn.Module):
    """Inverse-class-frequency weighted Ridge + Random Projection (remedy/control for Plan B).

    Minimizes sum_i (ref_count / n_{y_i}) ||h_i^T W_out - y_i^T||_2^2 + lambda ||W_out||_F^2.
    When all classes have `ref_count` samples, reduces identically to `Ridge`.
    """

    def __init__(self, feature_dim, num_classes=NUM_CLASSES, projection_dim=0,
                 seed=SEED, activation="relu", ref_count=500.0):
        super().__init__()
        self.activation = activation
        self.ref_count = float(ref_count)
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "W", torch.randn(feature_dim, projection_dim, generator=gen)
            if projection_dim else None,
        )
        dim = projection_dim or feature_dim
        self.register_buffer("G", torch.zeros(dim, dim))
        self.register_buffer("C", torch.zeros(dim, num_classes))
        self.register_buffer("counts", torch.zeros(num_classes, dtype=torch.long))

    def project(self, features):
        if self.W is None:
            return features
        projected = features @ self.W
        return apply_activation(projected, self.activation, inplace=True)

    def update(self, projected, labels):
        batch_counts = torch.bincount(labels, minlength=self.counts.numel())
        if ((self.counts > 0) & (batch_counts > 0)).any():
            raise ValueError("ClassBalancedRidge expects disjoint class arrivals per task (CIL).")
        self.counts += batch_counts
        per_class_n = self.counts[labels].to(projected.dtype).clamp_min(1.0)
        sqrt_w = torch.sqrt(self.ref_count / per_class_n).unsqueeze(1)
        weighted_h = projected * sqrt_w
        targets = F.one_hot(labels, num_classes=self.C.shape[1]).to(projected.dtype) * sqrt_w
        self.G.addmm_(weighted_h.T, weighted_h)
        self.C.addmm_(projected.T, F.one_hot(labels, num_classes=self.C.shape[1]).to(projected.dtype) * (sqrt_w * sqrt_w))

    def weights(self, seen_classes, ridge):
        resolution = self.G.diagonal().max().item() * torch.finfo(self.G.dtype).eps
        dtype = torch.float64 if ridge <= resolution else self.G.dtype
        system = self.G.to(dtype=dtype, copy=True)
        system.diagonal().add_(ridge)
        return torch.linalg.solve(system, self.C[:, :seen_classes].to(dtype)).to(self.C.dtype)

    def logits(self, features, weights):
        return self.project(features) @ weights


@torch.no_grad()
def evaluate_imbalance_diagnostics(classifier, weights, test_shards, class_counts, stats_device,
                                   batch_size=512):
    """Compute overall, head, tail accuracies, prediction share bias, and weight-norm ratio."""
    num_classes = len(class_counts)
    correct_per_class = torch.zeros(num_classes, dtype=torch.long)
    total_per_class = torch.zeros(num_classes, dtype=torch.long)
    pred_hist = torch.zeros(num_classes, dtype=torch.long)

    for features, labels in test_shards:
        for start in range(0, len(labels), batch_size):
            end = start + batch_size
            batch_x = features[start:end].to(stats_device)
            batch_y = labels[start:end].cpu()
            preds = classifier.logits(batch_x, weights).argmax(dim=1).cpu()
            pred_hist += torch.bincount(preds, minlength=num_classes)
            total_per_class += torch.bincount(batch_y, minlength=num_classes)
            hits = batch_y[preds == batch_y]
            if len(hits):
                correct_per_class += torch.bincount(hits, minlength=num_classes)

    per_class_acc = 100.0 * correct_per_class.float() / total_per_class.clamp_min(1).float()
    counts_t = torch.as_tensor(class_counts, dtype=torch.float32)
    max_n, min_n = float(counts_t.max().item()), float(counts_t.min().item())
    if max_n == min_n:
        # Balanced control: split Tasks 1..5 vs Tasks 6..10 as reference baseline groups.
        head_mask = torch.arange(num_classes) < (num_classes // 2)
        tail_mask = ~head_mask
    else:
        threshold = float((max_n * min_n) ** 0.5)
        head_mask = counts_t > threshold
        tail_mask = counts_t <= threshold

    head_acc = float(per_class_acc[head_mask].mean().item())
    tail_acc = float(per_class_acc[tail_mask].mean().item())
    col_norms = weights.detach().float().cpu().norm(dim=0)
    head_norm = float(col_norms[head_mask].mean().item())
    tail_norm = float(col_norms[tail_mask].mean().item())
    tail_pred_share = 100.0 * float(pred_hist[tail_mask].sum().item()) / max(1, int(pred_hist.sum().item()))
    expected_tail_share = 100.0 * float(tail_mask.float().mean().item())

    return {
        "a10_all_pct": float(per_class_acc.mean().item()),
        "head_acc_pct": head_acc,
        "tail_acc_pct": tail_acc,
        "head_tail_gap_pp": head_acc - tail_acc,
        "tail_pred_share_pct": tail_pred_share,
        "expected_tail_share_pct": expected_tail_share,
        "tail_pred_deficit_pp": tail_pred_share - expected_tail_share,
        "head_weight_norm": head_norm,
        "tail_weight_norm": tail_norm,
        "weight_norm_ratio_head_to_tail": head_norm / tail_norm if tail_norm > 0 else float("inf"),
        "per_class_acc_pct": per_class_acc.tolist(),
    }


def run_imbalance_experiment(method, mode, imbalance_ratio, train_shards_full, test_shards,
                             work_dir, *, projection_dim=10000, fixed_lambda=100.0,
                             stats_device="cpu", seed=SEED, run_label="imb-01",
                             entity=None, project="RanPAC-reproduction", wandb_mode="online"):
    """Run one (method, imbalance_mode, ratio) configuration over 10 tasks and log diagnostics."""
    _validate_run_label(run_label)
    if method not in {"ncm", "ridge", "ridge-rp", "cb-ridge-rp"}:
        raise ValueError(f"Unsupported imbalance method {method!r}.")

    ratio_tag = f"r{int(imbalance_ratio)}" if float(imbalance_ratio).is_integer() else f"r{imbalance_ratio}"
    lam_tag = "cv" if fixed_lambda is None else f"lam{int(fixed_lambda)}"
    run_name = f"imb-{mode}-{ratio_tag}-{method}-{lam_tag}-s{seed}-{run_label}"
    run_dir = Path(work_dir) / "runs" / run_name
    ckpt_path = run_dir / "checkpoint.pt"

    if ckpt_path.is_file():
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if saved.get("completed"):
            s = saved["summary"]
            print(f"[Đã xong] {run_name} -> All={s['a10_all_pct']:.2f}% | "
                  f"Head={s['head_acc_pct']:.2f}% | Tail={s['tail_acc_pct']:.2f}%")
            return saved

    counts = build_imbalance_counts(imbalance_ratio, mode=mode)
    subsampled_train = [
        subsample_task_shard(fx, fy, counts, seed=seed)
        for fx, fy in train_shards_full
    ]
    feature_dim = subsampled_train[0][0].shape[1]
    stats_dev = torch.device(stats_device)

    if method == "ncm":
        classifier = NCM(feature_dim, num_classes=NUM_CLASSES)
    elif method == "ridge":
        classifier = Ridge(feature_dim, num_classes=NUM_CLASSES, projection_dim=0, seed=seed)
    elif method == "ridge-rp":
        classifier = Ridge(feature_dim, num_classes=NUM_CLASSES,
                           projection_dim=projection_dim, seed=seed, activation="relu")
    else:
        classifier = ClassBalancedRidge(feature_dim, num_classes=NUM_CLASSES,
                                        projection_dim=projection_dim, seed=seed,
                                        activation="relu", ref_count=max(counts))
    classifier.to(stats_dev)

    config = {
        "experiment": "class_imbalance",
        "method": method,
        "imbalance_mode": mode,
        "imbalance_ratio": float(imbalance_ratio),
        "total_train_samples": int(sum(counts)),
        "min_class_samples": int(min(counts)),
        "max_class_samples": int(max(counts)),
        "M": projection_dim if "rp" in method else 0,
        "lambda_mode": "none" if method == "ncm" else ("task_cv" if fixed_lambda is None else "fixed"),
        "lambda_fixed": None if method == "ncm" else fixed_lambda,
        "seed": seed,
        "stats_device": stats_device,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb = None if wandb_mode == "disabled" else configure_wandb(entity, wandb_mode)
    wandb_id = uuid4().hex
    run_ctx = nullcontext()
    if wandb is not None:
        run_ctx = wandb.init(
            entity=entity, project=project, id=wandb_id, name=run_name,
            group=f"cifar100-class-imbalance-{run_label}",
            tags=["empirics", "critique", "class-imbalance", mode, method],
            config=config, mode=wandb_mode,
            dir=str(run_dir) if wandb_mode == "offline" else "/content",
        )

    records = []
    final_weights = None
    with run_ctx as run:
        if run is not None:
            run.define_metric("phase2/*", step_metric="phase2/task")
        for task_idx, (train_x, train_y) in enumerate(subsampled_train, start=1):
            t0 = perf_counter()
            seen_classes = task_idx * CLASSES_PER_TASK
            x_dev, y_dev = train_x.to(stats_dev), train_y.to(stats_dev)
            ridge_val = None
            if isinstance(classifier, (Ridge, ClassBalancedRidge)):
                projected = classifier.project(x_dev)
                ridge_val = fixed_lambda
                if ridge_val is None:
                    ridge_val = select_ridge(projected, y_dev, NUM_CLASSES, seed, task_idx)
                classifier.update(projected, y_dev)
                del projected
                weights = classifier.weights(seen_classes, ridge_val)
            else:
                classifier.update(x_dev, y_dev)
                weights = classifier.weights(seen_classes)
            del x_dev, y_dev

            accs = evaluate_seen(classifier, weights, test_shards[:task_idx], stats_dev)
            rec = {
                "task": task_idx,
                "seen_classes": seen_classes,
                "avg_accuracy_pct": sum(accs) / len(accs),
                "accuracies_pct": accs,
                "ridge_lambda": ridge_val,
                "task_seconds": perf_counter() - t0,
            }
            records.append(rec)
            final_weights = weights
            if run is not None:
                run.log({
                    "phase2/task": task_idx,
                    "phase2/avg_accuracy_pct": rec["avg_accuracy_pct"],
                    **({"phase2/ridge_lambda": ridge_val} if ridge_val is not None else {}),
                })

        diag = evaluate_imbalance_diagnostics(
            classifier, final_weights, test_shards, counts, stats_dev,
        )
        summary = {
            "method": method,
            "imbalance_mode": mode,
            "imbalance_ratio": float(imbalance_ratio),
            "lambda_mode": config["lambda_mode"],
            "total_train_samples": int(sum(counts)),
            "fit_eval_seconds": float(sum(r["task_seconds"] for r in records)),
            **{k: v for k, v in diag.items() if k != "per_class_acc_pct"},
            "wandb_url": run.url if run is not None else None,
        }
        if run is not None:
            for k, v in summary.items():
                if isinstance(v, (int, float)):
                    run.summary[k] = v

    payload = {
        "config": config,
        "counts": counts,
        "records": records,
        "per_class_acc_pct": diag["per_class_acc_pct"],
        "summary": summary,
        "completed": True,
    }
    save_checkpoint(ckpt_path, payload)
    export_results(run_dir, records)
    print(f"[{mode} ρ={imbalance_ratio:g} | {method:11s}] "
          f"All={summary['a10_all_pct']:.2f}% | Head={summary['head_acc_pct']:.2f}% | "
          f"Tail={summary['tail_acc_pct']:.2f}% | Gap={summary['head_tail_gap_pp']:+.2f} pp")
    return payload


# ============================================================================
# Plan C: Pre-trained Backbone Quality & Pre-training Objective Sensitivity
# ============================================================================


BACKBONE_SPECS = {
    "vit_b16_in21k": {
        "title": "ViT-B/16 · Sup. IN-21K (Chuẩn Paper)",
        "model_id": MODEL_ID,
        "checkpoint_url": CHECKPOINT_URL,
        "pretrain_regime": "Supervised ImageNet-21K (14M)",
        "feature_dim": 768,
        "use_paper_cache": True,
    },
    "vit_b16_in1k": {
        "title": "ViT-B/16 · Sup. IN-1K",
        "model_id": "vit_base_patch16_224.augreg_in1k",
        "checkpoint_url": None,
        "pretrain_regime": "Supervised ImageNet-1K (1.3M)",
        "feature_dim": 768,
        "use_paper_cache": False,
    },
    "vit_b16_dino": {
        "title": "ViT-B/16 · Self-Sup. DINO (IN-1K)",
        "model_id": "vit_base_patch16_224.dino",
        "checkpoint_url": None,
        "pretrain_regime": "Self-Supervised DINO IN-1K (No labels)",
        "feature_dim": 768,
        "use_paper_cache": False,
    },
    "vit_t16_in21k": {
        "title": "ViT-Ti/16 · Sup. IN-21K (5.5M params)",
        "model_id": "vit_tiny_patch16_224.augreg_in21k",
        "checkpoint_url": None,
        "pretrain_regime": "Supervised ImageNet-21K (Tiny, d=192)",
        "feature_dim": 192,
        "use_paper_cache": False,
    },
}


def load_or_extract_backbone_shards(backbone_key, work_dir, data_root, device, *,
                                    batch_size=64, workers=2, seed=SEED):
    """Return `(train_shards, test_shards, cache_dir)` for any registered backbone."""
    if backbone_key not in BACKBONE_SPECS:
        raise ValueError(f"Unknown backbone_key {backbone_key!r}. Choose from {tuple(BACKBONE_SPECS)}.")
    spec = BACKBONE_SPECS[backbone_key]
    class_order = split_class_order(seed)
    work_dir = Path(work_dir)

    if spec["use_paper_cache"]:
        cache = FeatureCache(
            work_dir / "embeddings", data_root,
            feature_config(class_order, torch.device("cuda")),
            device, batch_size, workers,
        )
        train_shards = [cache.load("train", t)[:2] for t in range(1, NUM_TASKS + 1)]
        test_shards = [cache.load("test", t)[:2] for t in range(1, NUM_TASKS + 1)]
        return train_shards, test_shards, cache.root

    key_cfg = {
        "dataset": "CIFAR-100",
        "class_order": class_order,
        "classes_per_task": CLASSES_PER_TASK,
        "backbone_key": backbone_key,
        "model_id": spec["model_id"],
        "feature_dim": spec["feature_dim"],
        "eval_transform": "timm_resolve_model_data_config",
    }
    digest = hashlib.sha256(json.dumps(key_cfg, sort_keys=True).encode()).hexdigest()[:16]
    shard_dir = work_dir / "embeddings" / f"{backbone_key}-{digest}"

    missing = [
        (sp, t)
        for t in range(1, NUM_TASKS + 1)
        for sp in ("train", "test")
        if not (shard_dir / f"{sp}-task-{t:02}.pt").is_file()
    ]
    if missing:
        if device.type != "cuda":
            raise RuntimeError(f"Shards missing for {backbone_key}; select a GPU runtime to extract.")
        import timm
        from ranpac_data import TaskImages
        from torchvision import datasets

        timm.layers.set_fused_attn(False)
        model = timm.create_model(spec["model_id"], pretrained=True, num_classes=0)
        model.eval().requires_grad_(False).to(device)
        data_cfg = timm.data.resolve_model_data_config(model)
        eval_tf = timm.data.create_transform(**data_cfg, is_training=False)

        ds_map = {
            sp: datasets.CIFAR100(root=data_root, train=(sp == "train"),
                                  transform=eval_tf, download=True)
            for sp in ("train", "test")
        }
        shard_dir.mkdir(parents=True, exist_ok=True)
        with atomic_output(shard_dir / "backbone_config.json") as tmp:
            tmp.write_text(json.dumps({**key_cfg, "timm_data_config": {
                k: list(v) if isinstance(v, tuple) else str(v) for k, v in data_cfg.items()
            }}, indent=2) + "\n", encoding="utf-8")

        for sp in ("train", "test"):
            for t in range(1, NUM_TASKS + 1):
                out_path = shard_dir / f"{sp}-task-{t:02}.pt"
                if out_path.is_file():
                    continue
                task_ds = TaskImages(ds_map[sp], class_order, t, CLASSES_PER_TASK)
                loader = DataLoader(
                    task_ds, batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=True,
                )
                batches = []
                with torch.inference_mode():
                    for images, _ in loader:
                        emb = model(images.to(device, non_blocking=True))
                        batches.append(emb.float().cpu())
                feats = torch.cat(batches, dim=0)
                if feats.shape != (len(task_ds), spec["feature_dim"]) or not torch.isfinite(feats).all():
                    raise ValueError(f"Invalid features for {backbone_key} {sp} task {t}: {feats.shape}")
                with atomic_output(out_path) as tmp:
                    torch.save((feats, torch.from_numpy(task_ds.labels)), tmp)
            print(f"[{backbone_key}] Đã trích xong toàn bộ 10 shard {sp}.")
        del model
        torch.cuda.empty_cache()

    train_shards = [
        torch.load(shard_dir / f"train-task-{t:02}.pt", map_location="cpu", weights_only=True)
        for t in range(1, NUM_TASKS + 1)
    ]
    test_shards = [
        torch.load(shard_dir / f"test-task-{t:02}.pt", map_location="cpu", weights_only=True)
        for t in range(1, NUM_TASKS + 1)
    ]
    return train_shards, test_shards, shard_dir


@torch.no_grad()
def compute_embedding_geometry(train_shards, num_classes=NUM_CLASSES):
    """Compute L2-normalized Fisher ratio Tr(S_B)/Tr(S_W) and angular prototype margin."""
    feats = F.normalize(torch.cat([x for x, _ in train_shards], dim=0).float(), dim=1)
    labels = torch.cat([y for _, y in train_shards], dim=0).long()

    global_mean = feats.mean(dim=0)
    class_means = torch.zeros(num_classes, feats.shape[1], dtype=torch.float32)
    counts = torch.zeros(num_classes, dtype=torch.float32)
    intra_cos_sum = 0.0
    for c in range(num_classes):
        cls_x = feats[labels == c]
        counts[c] = len(cls_x)
        mu_c = cls_x.mean(dim=0)
        class_means[c] = mu_c
        proto_c = F.normalize(mu_c, dim=0)
        intra_cos_sum += float((cls_x @ proto_c).sum().item())

    total_n = float(counts.sum().item())
    between_trace = float((counts[:, None] * (class_means - global_mean).pow(2)).sum().item()) / total_n
    centered = feats - class_means[labels]
    within_trace = float(centered.pow(2).sum().item()) / total_n
    fisher_trace_ratio = between_trace / within_trace if within_trace > 0 else float("inf")

    protos = F.normalize(class_means, dim=1)
    sim_matrix = protos @ protos.T
    sim_matrix.fill_diagonal_(-2.0)
    nearest_inter_cos = float(sim_matrix.max(dim=1).values.mean().item())
    mean_intra_cos = intra_cos_sum / total_n

    return {
        "feature_dim": int(feats.shape[1]),
        "mean_intra_cos": mean_intra_cos,
        "nearest_inter_cos": nearest_inter_cos,
        "angular_margin": mean_intra_cos - nearest_inter_cos,
        "fisher_trace_ratio": fisher_trace_ratio,
    }


def run_backbone_suite(backbone_key, work_dir, *, data_root="/content/ranpac-data",
                       projection_dim=10000, fixed_lambda=None, batch_size=64,
                       workers=2, stats_device="cuda", run_label="bb-01",
                       entity=None, project="RanPAC-reproduction", wandb_mode="online"):
    """Evaluate NCM, Ridge (no RP), and RanPAC Phase 2 (RP+ReLU) on one frozen backbone."""
    _validate_run_label(run_label)
    spec = BACKBONE_SPECS[backbone_key]
    feature_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats_dev_str = stats_device if (stats_device == "cpu" or torch.cuda.is_available()) else "cpu"

    train_shards, test_shards, _ = load_or_extract_backbone_shards(
        backbone_key, work_dir, data_root, feature_device,
        batch_size=batch_size, workers=workers, seed=SEED,
    )
    geometry = compute_embedding_geometry(train_shards, num_classes=NUM_CLASSES)

    results = {}
    task_sizes = (CLASSES_PER_TASK,) * NUM_TASKS
    for method in ("ncm", "ridge", "ridge-rp"):
        lam_tag = "none" if method == "ncm" else ("cv" if fixed_lambda is None else f"lam{int(fixed_lambda)}")
        run_name = f"bb-{backbone_key}-{method}-{lam_tag}-s{SEED}-{run_label}"
        run_dir = Path(work_dir) / "runs" / run_name
        ckpt_path = run_dir / "checkpoint.pt"

        if ckpt_path.is_file():
            saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if len(saved.get("records", ())) == NUM_TASKS:
                results[method] = saved
                print(f"[Đã xong] {run_name} -> A[10] = {saved['records'][-1]['avg_accuracy_pct']:.2f}%")
                continue

        config = {
            "experiment": "backbone_sensitivity",
            "backbone_key": backbone_key,
            "backbone_title": spec["title"],
            "model_id": spec["model_id"],
            "pretrain_regime": spec["pretrain_regime"],
            "feature_dim": spec["feature_dim"],
            "method": method,
            "M": projection_dim if method == "ridge-rp" else 0,
            "lambda_mode": "none" if method == "ncm" else ("task_cv" if fixed_lambda is None else "fixed"),
            "lambda_fixed": fixed_lambda if method != "ncm" else None,
            "seed": SEED,
            "stats_device": stats_dev_str,
            **geometry,
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        wandb = None if wandb_mode == "disabled" else configure_wandb(entity, wandb_mode)
        run_ctx = nullcontext()
        if wandb is not None:
            run_ctx = wandb.init(
                entity=entity, project=project, id=uuid4().hex, name=run_name,
                group=f"cifar100-backbone-sensitivity-{run_label}",
                tags=["empirics", "critique", "backbone-sensitivity", backbone_key, method],
                config=config, mode=wandb_mode,
                dir=str(run_dir) if wandb_mode == "offline" else "/content",
            )

        with run_ctx as run:
            if run is not None:
                run.define_metric("phase2/*", step_metric="phase2/task")
            records, _ = run_phase2_on_shards(
                train_shards, test_shards, task_sizes,
                method=method, projection_dim=projection_dim,
                fixed_lambda=fixed_lambda, stats_device=stats_dev_str, seed=SEED,
            )
            a10 = records[-1]["avg_accuracy_pct"]
            if run is not None:
                for rec in records:
                    run.log({
                        "phase2/task": rec["task"],
                        "phase2/avg_accuracy_pct": rec["avg_accuracy_pct"],
                        **({"phase2/ridge_lambda": rec["ridge_lambda"]}
                           if rec["ridge_lambda"] is not None else {}),
                    })
                run.summary["a10_pct"] = a10
                for gk, gv in geometry.items():
                    run.summary[gk] = gv
            saved = {
                "config": config,
                "records": records,
                "geometry": geometry,
                "wandb_url": run.url if run is not None else None,
            }
            save_checkpoint(ckpt_path, saved)
            export_results(run_dir, records)
            results[method] = saved
            print(f"[{backbone_key} | {method:8s}] A[10] = {a10:.2f}%")

    ncm_a10 = results["ncm"]["records"][-1]["avg_accuracy_pct"]
    ridge_a10 = results["ridge"]["records"][-1]["avg_accuracy_pct"]
    rp_a10 = results["ridge-rp"]["records"][-1]["avg_accuracy_pct"]
    err_ridge = 100.0 - ridge_a10
    rel_er = 100.0 * (ridge_a10 * 0 + (rp_a10 - ridge_a10)) / err_ridge if err_ridge > 0 else 0.0

    return {
        "backbone_key": backbone_key,
        "title": spec["title"],
        "pretrain_regime": spec["pretrain_regime"],
        "feature_dim": spec["feature_dim"],
        **geometry,
        "ncm_a10_pct": ncm_a10,
        "ridge_no_rp_a10_pct": ridge_a10,
        "ranpac_rp_a10_pct": rp_a10,
        "delta_rp_vs_ridge_pp": rp_a10 - ridge_a10,
        "rel_error_reduction_pct": rel_er,
        "runs": results,
    }
