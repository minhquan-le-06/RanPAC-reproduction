"""Algorithm 1 with a first-session AdaptFormer or SSF."""

import csv
import hashlib
import json
import math
from contextlib import nullcontext
from importlib.metadata import version
from itertools import chain
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ranpac_colab import configure_wandb, export_results, log_task, save_checkpoint, train_task
from ranpac_core import (
    CLASSES_PER_TASK, NUM_CLASSES, NUM_TASKS, RIDGE_GRID, SEED, NCM, Ridge, split_class_order,
)
from ranpac_data import (
    CHECKPOINT_URL, FEATURE_DIM, MODEL_ID, FeatureCache, TaskImages, atomic_output,
    feature_config,
)

BATCH_SIZE = 48
BOTTLENECK = 64
ADAPTER_SCALE = 0.1
ADAPTER_DROPOUT = 0.1
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005


class AdaptedBlock(nn.Module):
    """Add a bottleneck branch parallel to the frozen FFN of a timm ViT block."""

    def __init__(self, block, width):
        super().__init__()
        self.block = block.requires_grad_(False)
        down = nn.Linear(width, BOTTLENECK)
        up = nn.Linear(BOTTLENECK, width)
        nn.init.zeros_(down.bias)
        nn.init.zeros_(up.weight)
        nn.init.zeros_(up.bias)
        self.adapter = nn.Sequential(down, nn.ReLU(), nn.Dropout(ADAPTER_DROPOUT), up)

    def forward(self, tokens):
        block = self.block
        attention = block.attn(block.norm1(tokens))
        tokens = tokens + block.drop_path1(block.ls1(attention))
        ffn = block.drop_path2(block.ls2(block.mlp(block.norm2(tokens))))
        return tokens + (ffn + ADAPTER_SCALE * self.adapter(tokens))


class CosineHead(nn.Module):
    def __init__(self, num_classes=CLASSES_PER_TASK):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(num_classes, FEATURE_DIM))
        nn.init.uniform_(self.prototypes, -1 / math.sqrt(FEATURE_DIM), 1 / math.sqrt(FEATURE_DIM))
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, features):
        return self.scale * F.linear(F.normalize(features.float()), F.normalize(self.prototypes))


def build_adapted_backbone(device):
    import timm

    timm.layers.set_fused_attn(False)
    backbone = timm.create_model(
        MODEL_ID, pretrained=True, num_classes=0,
        pretrained_cfg_overlay={"hf_hub_id": None, "url": CHECKPOINT_URL},
    ).requires_grad_(False)
    backbone.blocks = nn.Sequential(*(AdaptedBlock(block, FEATURE_DIM) for block in backbone.blocks))
    return backbone.to(device)


def first_task_images(data_root, class_order, classes_per_task=CLASSES_PER_TASK, image_folder=None):
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3 / 4, 4 / 3)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = (datasets.CIFAR100(data_root, train=True, transform=transform, download=True)
               if image_folder is None else image_folder("train", transform))
    return TaskImages(dataset, class_order, task=1, classes_per_task=classes_per_task)


def train_epoch(backbone, head, images, optimizer, scaler, epoch, *, device, microbatch, workers):
    # Epoch-local RNG makes the data order and augmentation reproducible on resume.
    seed = SEED + epoch
    torch.manual_seed(seed)
    sampler_rng = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        images, batch_size=BATCH_SIZE, shuffle=True, generator=sampler_rng,
        num_workers=workers, pin_memory=device.type == "cuda",
    )
    backbone.train()
    head.train()
    started = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    for batch, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        for micro_images, micro_labels in zip(batch.split(microbatch), labels.split(microbatch)):
            truth = micro_labels.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                logits = head(backbone(micro_images.to(device, non_blocking=True)))
                loss = F.cross_entropy(logits, truth, reduction="sum")
            # Weight by sample count, including the final short batch.
            scaler.scale(loss / len(labels)).backward()
            loss_sum += loss.detach()
            correct += (logits.detach().argmax(1) == truth).sum()
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return {
        "epoch": epoch,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "train_loss": loss_sum.item() / len(images),
        "train_accuracy_pct": 100 * correct.item() / len(images),
        "epoch_seconds": perf_counter() - started,
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda" else None,
    }


def fit_first_task(backbone, adapters, checkpoint, config, last_epoch, data_root, device, commit,
                   image_folder=None):
    classes_per_task = config["classes_per_task"]
    head = (CosineHead(classes_per_task) if config["temporary_head"] == "cosine"
            else nn.Linear(FEATURE_DIM, classes_per_task))
    head.to(device)
    optimizer = torch.optim.SGD(
        chain(adapters.parameters(), head.parameters()), lr=LR, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["phase1_epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=config["precision"] == "amp-fp16")
    train_state = checkpoint["phase1_train_state"]
    if train_state is not None:
        head.load_state_dict(train_state["head"])
        optimizer.load_state_dict(train_state["optimizer"])
        scheduler.load_state_dict(train_state["scheduler"])
        scaler.load_state_dict(train_state["scaler"])

    images = first_task_images(data_root, config["class_order"], classes_per_task, image_folder)
    for epoch in range(len(checkpoint["epochs"]) + 1, last_epoch + 1):
        record = train_epoch(
            backbone, head, images, optimizer, scaler, epoch, device=device,
            microbatch=config["microbatch"], workers=config["workers"],
        )
        scheduler.step()
        checkpoint["adapter"] = adapters.state_dict()
        checkpoint["phase1_train_state"] = {
            "head": head.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        } if epoch < config["phase1_epochs"] else None
        checkpoint["epochs"].append(record)
        commit("phase1", record)
        print(f"Epoch {epoch:02}/{config['phase1_epochs']} | loss = {record['train_loss']:.4f} "
              f"| train acc = {record['train_accuracy_pct']:.2f}% | {record['epoch_seconds']:.1f}s")


def adapter_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def log_record(run, phase, record, step):
    payload = {f"{phase}/{key}": value for key, value in record.items()
               if key != "accuracies_pct" and value is not None}
    run.log({"checkpoint_step": step, **payload}, step=step, commit=True)


def run_algorithm1(work_dir, *, run_label="main", resume=False, phase1_epochs=20,
                   stop_after_epoch=None, stop_after_task=10, microbatch=48, precision="fp32",
                   temporary_head="cosine", projection_dim=10000, fixed_lambda=None,
                   feature_batch_size=48, stats_device="cpu", workers=2,
                   data_root="/content/ranpac-data", entity=None, project="RanPAC-reproduction",
                   wandb_mode="disabled", imagenetr_root=None, extra_tags=None):
    """Checkpoint after each Phase 1 epoch and each Phase 2 task on one run."""
    num_classes, classes_per_task = NUM_CLASSES, CLASSES_PER_TASK
    if imagenetr_root is not None:
        import ranpac_imagenetr as imagenetr

        num_classes, classes_per_task = imagenetr.NUM_CLASSES, imagenetr.CLASSES_PER_TASK
    num_tasks = num_classes // classes_per_task
    last_epoch = phase1_epochs if stop_after_epoch is None else stop_after_epoch
    if not (1 <= last_epoch <= phase1_epochs and 1 <= stop_after_task <= num_tasks):
        raise ValueError("Check the requested stop epoch and task.")
    if not 1 <= microbatch <= BATCH_SIZE or feature_batch_size < 1 or workers < 0:
        raise ValueError("Check microbatch, feature_batch_size and workers.")
    if (projection_dim < 1 or (fixed_lambda is not None and
                              (not math.isfinite(fixed_lambda) or fixed_lambda <= 0))):
        raise ValueError("M and optionally fixed lambda must be positive and finite.")
    if precision not in {"fp32", "amp-fp16"} or temporary_head not in {"cosine", "linear"}:
        raise ValueError("Check precision and temporary_head.")
    if stats_device not in {"cpu", "cuda"} or wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("Check stats_device and wandb_mode.")
    if not run_label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run_label):
        raise ValueError("Use lowercase letters, digits, '-' or '_' in run_label.")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab before running Algorithm 1.")

    device = torch.device("cuda")
    stats_device = torch.device(stats_device)
    torch.set_float32_matmul_precision("highest")
    image_folder = None
    petl = "adaptformer"
    dataset_tag = "cifar100"
    source_files = ["ranpac_core.py", "ranpac_data.py", "ranpac_colab.py", "ranpac_full.py"]
    data_config = feature_config(split_class_order(), device)
    if imagenetr_root is not None:
        image_data = imagenetr.ImageNetR(imagenetr_root)
        image_folder = image_data.image_folder
        petl = "ssf"
        dataset_tag = "imagenetr"
        source_files.append("ranpac_imagenetr.py")
        data_config.update(
            dataset="ImageNet-R",
            class_order=split_class_order(num_classes=num_classes),
            classes_per_task=classes_per_task,
            model_id=imagenetr.MODEL_ID,
            checkpoint=imagenetr.CHECKPOINT_URL,
            transforms="PIL Resize(256, BICUBIC) -> CenterCrop(224) -> ToTensor()",
            split_manifest_sha256=image_data.manifest_sha256,
        )
    wandb = None if wandb_mode == "disabled" else configure_wandb(entity, wandb_mode)
    source_digest = hashlib.sha256()
    for name in source_files:
        source_digest.update(Path(__file__).with_name(name).read_bytes())
    config = {
        **data_config, "seed": SEED, "petl": petl, "phase1_epochs": phase1_epochs,
        "phase1_train_transforms": "RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3/4, 4/3)) -> RandomHorizontalFlip(0.5) -> ToTensor()",
        "effective_batch_size": BATCH_SIZE, "microbatch": microbatch, "precision": precision,
        "temporary_head": temporary_head, "optimizer": "sgd", "lr": LR, "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY, "scheduler": "cosine",
        **({"adapter_bottleneck": BOTTLENECK, "adapter_scale": ADAPTER_SCALE,
            "adapter_dropout": ADAPTER_DROPOUT} if petl == "adaptformer" else {"ssf_init_std": 0.02}),
        "M": projection_dim, "lambda_mode": "task_cv" if fixed_lambda is None else "fixed",
        "lambda_grid": list(RIDGE_GRID) if fixed_lambda is None else None,
        "lambda_fixed": fixed_lambda, "stats_device": str(stats_device), "stats_dtype": "float32",
        "feature_batch_size": feature_batch_size, "workers": workers,
        "source_sha256": source_digest.hexdigest(),
        "wandb": {"mode": wandb_mode, "entity": entity, "project": project,
                  "version": version("wandb") if wandb is not None else None},
    }
    work_dir = Path(work_dir)
    run_prefix = "full" if imagenetr_root is None else "full-imagenetr"
    run_name = f"{run_prefix}-s{SEED}-{run_label}"
    run_dir = work_dir / "runs" / run_name
    checkpoint_path = run_dir / "checkpoint.pt"
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Existing CIFAR-100 checkpoints retain their feature-cache identity.
        previous_digest = checkpoint["config"].get("source_sha256")
        if previous_digest in {
            "ceb50cc1e96e9b1da60ca140c9c1728557a1f705779c67c94465622fded71c49",
            "1a26a0627426b68b81986a9d768b851be8316329d7da1f9395dbfe21752cc033",
            "28490f2a7df166bcd40f07c1d14a304d30d5da051f18daacf5c4f694872bfb70",
            "d0b67c15cfc9c7c8daccdecaacb4ce7566cead058a37c17f45af2380ca2fcddb",
            "322e636cc1de59ad1ebad019b5c6df7d5575cc51af5ec067273dfb292bc65982",
            "22b449fab225c330002251c8f56bbfdba61e7204dab2b37c9c46ef0020c40d25",
        }:
            config["source_sha256"] = previous_digest
        if checkpoint["config"] != config:
            raise ValueError("Config differs from the checkpoint. Restore it or use a new run label.")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = {
            "config": config, "wandb_id": uuid4().hex, "wandb_initialized": False,
            "adapter": None, "phase1_train_state": None, "ridge": None, "epochs": [], "tasks": [],
        }
        save_checkpoint(checkpoint_path, checkpoint)
    epochs, tasks = checkpoint["epochs"], checkpoint["tasks"]
    if last_epoch < len(epochs) or stop_after_task < len(tasks):
        raise ValueError("The requested stop precedes a completed epoch or task.")
    with atomic_output(run_dir / "config.json") as temp:
        temp.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    def export_metrics():
        if epochs:
            with atomic_output(run_dir / "phase1_metrics.csv") as temp, temp.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(epochs[0]))
                writer.writeheader()
                writer.writerows(epochs)
        export_results(run_dir, tasks)

    def commit(phase, record):
        save_checkpoint(checkpoint_path, checkpoint)
        export_metrics()
        if run is not None:
            log_record(run, phase, record, len(epochs) + len(tasks))

    def run_phases():
        if len(epochs) == last_epoch and (last_epoch < phase1_epochs or len(tasks) == stop_after_task):
            return
        torch.manual_seed(SEED)
        if imagenetr_root is None:
            backbone = build_adapted_backbone(device)
            adapters = nn.ModuleList(block.adapter for block in backbone.blocks)
        else:
            backbone, adapters = imagenetr.build_ssf_backbone(device)
        if checkpoint["adapter"] is not None:
            adapters.load_state_dict(checkpoint["adapter"])
        if len(epochs) < last_epoch:
            fit_first_task(backbone, adapters, checkpoint, config, last_epoch, data_root, device,
                           commit, image_folder)
        if len(epochs) < phase1_epochs:
            return

        backbone.eval().requires_grad_(False)
        cache_config = {
            **data_config, "adapter_sha256": adapter_sha256(checkpoint["adapter"]),
            "implementation_sha256": config["source_sha256"],
        }
        cache = FeatureCache(work_dir / "embeddings", data_root, cache_config, device,
                             feature_batch_size, workers, image_folder=image_folder)
        cache.model = backbone
        classifier = Ridge(FEATURE_DIM, num_classes=num_classes, projection_dim=projection_dim, seed=SEED)
        if checkpoint["ridge"] is not None:
            classifier.load_state_dict(checkpoint["ridge"], assign=True)
            checkpoint["ridge"] = None
        classifier.to(stats_device)
        for task in range(len(tasks) + 1, stop_after_task + 1):
            record = train_task(classifier, cache, task, stats_device, device, fixed_lambda,
                                classes_per_task=classes_per_task, num_classes=num_classes)
            tasks.append(record)
            checkpoint["ridge"] = classifier.state_dict()
            commit("phase2", record)
            print(f"Task {task:02}/{num_tasks} | A[{task}] = {record['avg_accuracy_pct']:.2f}% "
                  f"| {record['task_seconds']:.1f}s")

    print(f"Results: {run_dir}")
    export_metrics()
    run_context = nullcontext()
    if wandb is not None:
        resume_option = None
        if wandb_mode == "online":
            resume_option = "must" if checkpoint["wandb_initialized"] else "allow" if resume else "never"
        tags = list(dict.fromkeys([petl, *(extra_tags or ())]))
        run_context = wandb.init(
            entity=entity, project=project, id=checkpoint["wandb_id"], name=run_name,
            group=f"{dataset_tag}-algorithm1", tags=tags, config=config, mode=wandb_mode,
            resume=resume_option, dir=str(run_dir) if wandb_mode == "offline" else "/content",
        )
    with run_context as run:
        if run is not None:
            print(f"W&B run {run.id}: {run.url or run.dir}")
            run.define_metric("phase1/*", step_metric="phase1/epoch")
            run.define_metric("phase2/*", step_metric="phase2/task")
            if not checkpoint["wandb_initialized"]:
                checkpoint["wandb_initialized"] = True
                save_checkpoint(checkpoint_path, checkpoint)
            if resume and wandb_mode == "online" and (epochs or tasks):
                history = wandb.Api().run(f"{entity}/{project}/{run.id}").scan_history(keys=["checkpoint_step"])
                logged = max((int(row["checkpoint_step"]) for row in history), default=0)
                if logged > len(epochs) + len(tasks):
                    raise ValueError("W&B is ahead of the checkpoint. Check the Drive run directory.")
                events = chain((("phase1", row) for row in epochs), (("phase2", row) for row in tasks))
                for step, (phase, row) in enumerate(events, start=1):
                    if step > logged:
                        log_record(run, phase, row, step)
        run_phases()
        if len(tasks) == num_tasks:
            accuracy = tasks[-1]["avg_accuracy_pct"]
            if run is not None:
                run.summary["a10_pct"] = accuracy
            print(f"Final A[10] = {accuracy:.2f}%")
    return {"epochs": epochs, "tasks": tasks, "run_dir": run_dir}


PHASE1_ABLATION_PRESETS = {
    "phase1-ncm": ("ncm", 0, None, 87.8),       # Table 1: NCM with Phase 1
    "phase1-no-rp": ("ridge", 0, None, 90.6),   # Table 1: No RPs (Phase 1 + Ridge 768-d)
}


def run_phase1_ablation(preset, work_dir, *, adapter_checkpoint_path=None, run_label="main",
                        resume=False, stop_after_task=NUM_TASKS, feature_batch_size=48,
                        stats_device="cpu", workers=2, data_root="/content/ranpac-data",
                        entity=None, project="RanPAC-reproduction", wandb_mode="online",
                        extra_tags=None):
    """Run Table 1 Phase 2 ablations ('No RPs', 'NCM with Phase 1') reusing a trained Phase 1 adapter."""
    if preset not in PHASE1_ABLATION_PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from {tuple(PHASE1_ABLATION_PRESETS)}.")
    if not 1 <= stop_after_task <= NUM_TASKS:
        raise ValueError(f"Set stop_after_task between 1 and {NUM_TASKS}.")
    if not run_label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run_label):
        raise ValueError("Use lowercase letters, digits, '-' or '_' in run_label.")
    if stats_device not in {"cpu", "cuda"} or wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("Check stats_device and wandb_mode.")

    work_dir = Path(work_dir)
    source_ckpt_path = Path(
        adapter_checkpoint_path or (work_dir / "runs" / f"full-s{SEED}-main" / "checkpoint.pt")
    )
    if not source_ckpt_path.is_file():
        raise FileNotFoundError(
            f"Phase 1 checkpoint not found at {source_ckpt_path}. "
            "Run 04_ranpac_full.ipynb first or pass adapter_checkpoint_path."
        )
    source_ckpt = torch.load(source_ckpt_path, map_location="cpu", weights_only=True)
    source_cfg = source_ckpt["config"]
    if source_ckpt.get("adapter") is None or len(source_ckpt.get("epochs", ())) < source_cfg["phase1_epochs"]:
        raise ValueError(f"Checkpoint {source_ckpt_path} has not completed Phase 1 training.")

    method, projection_dim, fixed_lambda, paper_accuracy = PHASE1_ABLATION_PRESETS[preset]
    feature_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats_device = torch.device(stats_device)
    torch.set_float32_matmul_precision("highest")
    # Match feature_device_type='cuda' used by 04_ranpac_full.ipynb so cached shards hit directly.
    data_config = feature_config(split_class_order(), torch.device("cuda"))
    adapter_digest = adapter_sha256(source_ckpt["adapter"])
    cache_config = {
        **data_config,
        "adapter_sha256": adapter_digest,
        "implementation_sha256": source_cfg["source_sha256"],
    }
    cache = FeatureCache(work_dir / "embeddings", data_root, cache_config, feature_device,
                         feature_batch_size, workers)

    missing_shards = [
        (split, task)
        for task in range(1, stop_after_task + 1)
        for split in ("train", "test")
        if not (cache.root / f"{split}-task-{task:02}.pt").is_file()
    ]
    if missing_shards:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Adapted feature shards missing in {cache.root}; select a GPU runtime to extract them."
            )
        backbone = build_adapted_backbone(feature_device)
        adapters = nn.ModuleList(block.adapter for block in backbone.blocks)
        adapters.load_state_dict(source_ckpt["adapter"])
        backbone.eval().requires_grad_(False)
        cache.model = backbone

    wandb = None if wandb_mode == "disabled" else configure_wandb(entity, wandb_mode)
    source_digest = hashlib.sha256()
    for name in ("ranpac_core.py", "ranpac_data.py", "ranpac_colab.py", "ranpac_full.py"):
        source_digest.update(Path(__file__).with_name(name).read_bytes())

    config = {
        **data_config,
        "preset": preset,
        "method": method,
        "seed": SEED,
        "petl": "adaptformer",
        "phase1_source_checkpoint": str(source_ckpt_path),
        "adapter_sha256": adapter_digest,
        "M": projection_dim,
        "lambda_mode": "none" if method == "ncm" else "task_cv" if fixed_lambda is None else "fixed",
        "lambda_fixed": fixed_lambda,
        "lambda_grid": list(RIDGE_GRID) if method != "ncm" and fixed_lambda is None else None,
        "stats_device": str(stats_device),
        "stats_dtype": "float32",
        "feature_batch_size": feature_batch_size,
        "wandb_version": version("wandb") if wandb is not None else None,
        "source_sha256": source_digest.hexdigest(),
        "reference_table": "1",
        "paper_a10_pct": paper_accuracy,
    }
    run_name = f"{preset}-s{SEED}-{run_label}"
    run_dir = work_dir / "runs" / run_name
    checkpoint_path = run_dir / "checkpoint.pt"
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
    with atomic_output(run_dir / "config.json") as temp:
        temp.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    classifier = NCM(FEATURE_DIM) if method == "ncm" else Ridge(FEATURE_DIM, projection_dim=projection_dim)
    if checkpoint["model"] is not None:
        classifier.load_state_dict(checkpoint["model"])
        checkpoint["model"] = None
    classifier.to(stats_device)

    run_context = nullcontext()
    if wandb is not None:
        log_dir = run_dir if wandb_mode == "offline" else Path("/content")
        resume_option = None
        if wandb_mode == "online":
            resume_option = "must" if identity["initialized"] else "allow" if resume else "never"
        tags = list(dict.fromkeys(["adaptformer", f"phase1-{method}", *(extra_tags or ("reproduction",))]))
        run_context = wandb.init(
            entity=entity, project=project, id=identity["id"], name=run_name,
            group="cifar100-phase1-table-1", tags=tags, config=config,
            mode=wandb_mode, resume=resume_option, dir=str(log_dir),
        )

    with run_context as run:
        if run is not None:
            if not identity["initialized"] or (wandb_mode == "online" and not identity["url"]):
                identity.update(url=run.url, initialized=True)
                if records:
                    checkpoint["model"] = classifier.state_dict()
                save_checkpoint(checkpoint_path, checkpoint)
            run.define_metric("phase2/*", step_metric="phase2/task")
            print(f"W&B run {run.id}: {run.url or Path(run.dir).parent}")
        print(f"Results: {run_dir}")
        export_results(run_dir, records)

        for task in range(len(records) + 1, stop_after_task + 1):
            record = train_task(classifier, cache, task, stats_device, feature_device, fixed_lambda)
            record["cumulative_ridge_solve_seconds"] = sum(
                previous.get("ridge_solve_seconds") or 0 for previous in records
            ) + (record.get("ridge_solve_seconds") or 0)
            records.append(record)
            checkpoint["model"] = classifier.state_dict()
            if run is not None:
                identity["url"] = run.url or identity["url"]
            save_checkpoint(checkpoint_path, checkpoint)
            export_results(run_dir, records)
            if run is not None:
                log_task(wandb, run, records)
            print(f"Task {task:02}/{NUM_TASKS} | A[{task}] = {record['avg_accuracy_pct']:.2f}% "
                  f"| {record['task_seconds']:.1f}s")

        if len(records) == NUM_TASKS:
            result = records[-1]["avg_accuracy_pct"]
            if run is not None:
                run.summary["a10_pct"] = result
                run.summary["paper_a10_pct"] = paper_accuracy
                run.summary["delta_to_paper_pp"] = result - paper_accuracy
            print(f"Table 1 ({preset}) | A[10] {result:.2f}% | "
                  f"paper {paper_accuracy:.1f}% | Δ {result - paper_accuracy:+.2f} pp")
    return records
