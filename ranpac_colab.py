"""Run and resume the Phase 2 benchmarks from the three Colab notebooks."""

import csv
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import torch

from ranpac_core import (
    CLASSES_PER_TASK, NUM_CLASSES, NUM_TASKS, RIDGE_GRID, SEED, NCM, Ridge,
    evaluate_seen, select_ridge, split_class_order,
)
from ranpac_data import FEATURE_DIM, FeatureCache, atomic_output, feature_config

try:
    import resource
except ImportError:
    resource = None


# Method, RP dimension, fixed lambda (None means task CV for ridge), paper A[10].
PRESETS = {
    "ncm": ("ncm", 0, None, 83.4),
    "ridge": ("ridge", 0, None, 87.0),
    "rp-table1": ("ridge-rp", 10000, None, 89.0),
    "rp-table-a5": ("ridge-rp", 2500, 100.0, 87.7),
}


def configure_wandb(entity, mode):
    if not entity or mode not in {"online", "offline"}:
        raise ValueError("Set the W&B entity and select online or offline mode.")
    os.environ["WANDB_DISABLE_CODE"] = "true"
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        from google.colab import userdata

        os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")

    import wandb

    return wandb


def smoke_wandb(entity, project):
    wandb = configure_wandb(entity, "online")
    with wandb.init(entity=entity, project=project, name="smoke-colab", mode="online",
                    config={"purpose": "smoke"}) as run:
        run.log({"smoke/ok": 1})
        print(run.url)


def save_checkpoint(path, checkpoint):
    with atomic_output(path) as temp:
        torch.save(checkpoint, temp)


def accuracy_rows(records):
    return [(record["task"], test_task, accuracy)
            for record in records
            for test_task, accuracy in enumerate(record["accuracies_pct"], start=1)]


def export_results(run_dir, records):
    if not records:
        return
    metrics = [{key: value for key, value in row.items() if key != "accuracies_pct"}
               for row in records]
    for name, header, rows in (
        ("metrics.csv", list(metrics[0]), [list(row.values()) for row in metrics]),
        ("accuracy_matrix.csv", ["train_task", "test_task", "accuracy_pct"], accuracy_rows(records)),
    ):
        with atomic_output(run_dir / name) as temp, temp.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)


def log_task(wandb, run, records):
    record = records[-1]
    payload = {f"phase2/{key}": value for key, value in record.items()
               if key != "accuracies_pct" and value is not None}
    payload["results/accuracy_matrix"] = wandb.Table(
        columns=["train_task", "test_task", "accuracy_pct"], data=accuracy_rows(records),
    )
    run.log(payload, step=record["task"], commit=True)


def train_task(classifier, cache, task, stats_device, feature_device, fixed_lambda):
    started = perf_counter()
    if feature_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(feature_device)
    features, labels, feature_seconds = cache.load("train", task)
    seen_classes = task * CLASSES_PER_TASK
    fit_started = perf_counter()
    features, labels = features.to(stats_device), labels.to(stats_device)
    ridge = lambda_seconds = solve_seconds = None
    if isinstance(classifier, Ridge):
        projected = classifier.project(features)
        ridge = fixed_lambda
        if ridge is None:
            if stats_device.type == "cuda":
                torch.cuda.synchronize(stats_device)
            search_started = perf_counter()
            ridge = select_ridge(projected, labels, NUM_CLASSES, SEED, task)
            lambda_seconds = perf_counter() - search_started
        classifier.update(projected, labels)
        del projected
        if stats_device.type == "cuda":
            torch.cuda.synchronize(stats_device)
        solve_started = perf_counter()
        weights = classifier.weights(seen_classes, ridge)
        if stats_device.type == "cuda":
            torch.cuda.synchronize(stats_device)
        solve_seconds = perf_counter() - solve_started
    else:
        classifier.update(features, labels)
        weights = classifier.weights(seen_classes)
        if stats_device.type == "cuda":
            torch.cuda.synchronize(stats_device)
    fit_seconds = perf_counter() - fit_started
    del features, labels

    test_shards = []
    for test_task in range(1, task + 1):
        test_features, test_labels, seconds = cache.load("test", test_task)
        test_shards.append((test_features, test_labels))
        feature_seconds += seconds
    eval_started = perf_counter()
    accuracies = evaluate_seen(classifier, weights, test_shards, stats_device)
    return {
        "task": task,
        "seen_classes": seen_classes,
        "avg_accuracy_pct": sum(accuracies) / len(accuracies),
        "accuracies_pct": accuracies,
        "ridge_lambda": ridge,
        "lambda_search_seconds": lambda_seconds,
        "ridge_solve_seconds": solve_seconds,
        "feature_seconds": feature_seconds,
        "fit_seconds": fit_seconds,
        "eval_seconds": perf_counter() - eval_started,
        "task_seconds": perf_counter() - started,
        # Linux reports ru_maxrss in KiB. This is the process high-water mark.
        "host_peak_rss_mib": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 if resource else None
        ),
        "gpu_peak_allocated_mib": (
            torch.cuda.max_memory_allocated(feature_device) / 2**20
            if feature_device.type == "cuda" else None
        ),
    }


def run_phase2(preset, work_dir, *, entity, project="ranpac-cifar100", stop_after_task=2,
               run_label="main", resume=False, wandb_mode="online", batch_size=48,
               stats_device="cpu", workers=2, data_root="/content/ranpac-data"):
    if not 1 <= stop_after_task <= NUM_TASKS:
        raise ValueError(f"Set stop_after_task between 1 and {NUM_TASKS}.")
    if not run_label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run_label):
        raise ValueError("Use lowercase letters, digits, '-' or '_' in run_label.")
    if stats_device not in {"cpu", "cuda"} or batch_size < 1 or workers < 0:
        raise ValueError("Check stats_device, batch_size and workers.")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab before running the benchmark.")

    wandb = configure_wandb(entity, wandb_mode)
    torch.set_float32_matmul_precision("highest")
    feature_device = torch.device("cuda")
    stats_device = torch.device(stats_device)
    method, projection_dim, fixed_lambda, paper_accuracy = PRESETS[preset]
    data_config = feature_config(split_class_order(), feature_device)
    source_digest = hashlib.sha256()
    for name in ("ranpac_core.py", "ranpac_data.py", "ranpac_colab.py"):
        source_digest.update(Path(__file__).with_name(name).read_bytes())
    config = {
        **data_config,
        "preset": preset,
        "method": method,
        "seed": SEED,
        "petl": False,
        "M": projection_dim,
        "lambda_mode": "none" if method == "ncm" else "task_cv" if fixed_lambda is None else "fixed",
        "lambda_fixed": fixed_lambda,
        "lambda_grid": list(RIDGE_GRID) if method != "ncm" and fixed_lambda is None else None,
        "stats_device": str(stats_device),
        "stats_dtype": "float32",
        "feature_batch_size": batch_size,
        "wandb_version": version("wandb"),
        "source_sha256": source_digest.hexdigest(),
        "reference_table": "A5" if preset == "rp-table-a5" else "1",
        "paper_a10_pct": paper_accuracy,
    }
    run_name = f"{preset}-s{SEED}-{run_label}"
    work_dir = Path(work_dir)
    run_dir = work_dir / "runs" / run_name
    checkpoint_path = run_dir / "checkpoint.pt"
    cache = FeatureCache(work_dir / "embeddings", data_root, data_config, feature_device,
                         batch_size, workers)
    config_path = run_dir / "config.json"

    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        identity = checkpoint["wandb"]
        if checkpoint["config"] != config or (identity["entity"], identity["project"]) != (entity, project):
            raise ValueError("Config or W&B project differs from the checkpoint. Use a new run label.")
    else:
        identity = {"entity": entity, "project": project, "id": uuid4().hex,
                    "url": None, "initialized": False}
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = {"config": config, "wandb": identity, "model": None, "records": []}
        save_checkpoint(checkpoint_path, checkpoint)

    records = checkpoint["records"]
    if stop_after_task < len(records):
        raise ValueError("stop_after_task is earlier than the last completed task in the checkpoint.")
    with atomic_output(config_path) as temp:
        temp.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    classifier = NCM(FEATURE_DIM) if method == "ncm" else Ridge(FEATURE_DIM, projection_dim=projection_dim)
    if checkpoint["model"] is not None:
        classifier.load_state_dict(checkpoint["model"])
        checkpoint["model"] = None
    classifier.to(stats_device)

    # For offline continuation, preserve each SDK log directory on Drive for later sync.
    # Use run_dir for both offline and online when /content (Colab) doesn't exist.
    _colab_dir = Path("/content")
    log_dir = run_dir if wandb_mode == "offline" or not _colab_dir.exists() else _colab_dir
    resume_option = None
    if wandb_mode == "online":
        resume_option = "must" if (resume and identity["initialized"]) else "never"
    with wandb.init(
        entity=entity, project=project, id=identity["id"], name=run_name,
        group=f"cifar100-phase2-table-{config['reference_table']}", tags=[method],
        config=config, mode=wandb_mode, resume=resume_option, dir=str(log_dir),
    ) as run:
        if not identity["initialized"] or (wandb_mode == "online" and not identity["url"]):
            identity.update(url=run.url, initialized=True)
            # Existing checkpoint already holds model state when switching modes.
            if records:
                checkpoint["model"] = classifier.state_dict()
            save_checkpoint(checkpoint_path, checkpoint)
        run.define_metric("phase2/*", step_metric="phase2/task")
        run_location = run.url or Path(run.dir).parent
        print(f"W&B run {run.id}: {run_location}\nResults: {run_dir}")
        export_results(run_dir, records)

        if resume and records and wandb_mode == "online":
            # Checkpoints are written before logging. Replay any missing trailing task.
            history = wandb.Api().run(f"{entity}/{project}/{run.id}").scan_history(keys=["phase2/task"])
            last_logged = max((int(row["phase2/task"]) for row in history), default=0)
            if last_logged > len(records):
                raise ValueError("W&B is ahead of the checkpoint. Check the Drive run directory.")
            for task in range(last_logged + 1, len(records) + 1):
                log_task(wandb, run, records[:task])

        for task in range(len(records) + 1, stop_after_task + 1):
            record = train_task(classifier, cache, task, stats_device, feature_device, fixed_lambda)
            records.append(record)
            checkpoint["model"] = classifier.state_dict()
            identity["url"] = run.url or identity["url"]
            save_checkpoint(checkpoint_path, checkpoint)
            export_results(run_dir, records)
            log_task(wandb, run, records)
            print(f"Task {task:02}/{NUM_TASKS} | A[{task}] = {record['avg_accuracy_pct']:.2f}% "
                  f"| {record['task_seconds']:.1f}s")

        if len(records) == NUM_TASKS:
            result = records[-1]["avg_accuracy_pct"]
            run.summary["a10_pct"] = result
            run.summary["paper_a10_pct"] = paper_accuracy
            run.summary["delta_to_paper_pp"] = result - paper_accuracy
            print(f"Table {config['reference_table']} | A[10] {result:.2f}% | "
                  f"paper {paper_accuracy:.1f}% | Δ {result - paper_accuracy:+.2f} pp")
    return records
