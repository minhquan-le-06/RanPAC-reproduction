"""CIFAR-100 loading and shared ViT feature shards for Colab."""

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ranpac_core import CLASSES_PER_TASK, NUM_CLASSES


MODEL_ID = "vit_base_patch16_224.augreg_in21k"
CHECKPOINT_URL = (
    "https://storage.googleapis.com/vit_models/augreg/"
    "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz"
)
FEATURE_DIM = 768


@contextmanager
def atomic_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        yield Path(temp)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def feature_config(class_order, device):
    return {
        "dataset": "CIFAR-100",
        "class_order": class_order,
        "classes_per_task": CLASSES_PER_TASK,
        "model_id": MODEL_ID,
        "checkpoint": CHECKPOINT_URL,
        "transforms": "PIL Resize(224, BICUBIC) -> CenterCrop(224) -> ToTensor()",
        "input_normalization": False,
        "feature_dim": FEATURE_DIM,
        "feature_dtype": "float32",
        "feature_device_type": device.type,
        "fused_attention": False,
        "feature_libraries": {p: version(p) for p in ("torch", "torchvision", "timm", "Pillow")},
    }


class TaskImages(Dataset):
    def __init__(self, cifar, class_order, task):
        old_to_new = np.empty(NUM_CLASSES, dtype=np.int64)
        old_to_new[class_order] = np.arange(NUM_CLASSES)
        labels = old_to_new[np.asarray(cifar.targets)]
        stop = task * CLASSES_PER_TASK
        self.indices = np.flatnonzero((labels >= stop - CLASSES_PER_TASK) & (labels < stop))
        self.labels = labels[self.indices]
        self.cifar = cifar

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.cifar[int(self.indices[index])]
        return image, int(self.labels[index])


class FeatureCache:
    def __init__(self, root, data_root, config, device, batch_size, workers):
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
        self.root = Path(root) / key
        self.data_root = data_root
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.workers = workers
        self.datasets = {}
        self.model = None

    def load(self, split, task):
        path = self.root / f"{split}-task-{task:02}.pt"
        if path.exists():
            features, labels = torch.load(path, map_location="cpu", weights_only=True)
            return features, labels, 0.0

        started = perf_counter()
        features, labels = self.extract(split, task)
        with atomic_output(path) as temp:
            torch.save((features, labels), temp)
        return features, labels, perf_counter() - started

    def extract(self, split, task):
        import timm
        from torchvision import datasets, transforms

        if split not in self.datasets:
            transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
            self.datasets[split] = datasets.CIFAR100(
                root=self.data_root, train=split == "train", transform=transform, download=True,
            )

        if self.model is None:
            timm.layers.set_fused_attn(False)
            self.model = timm.create_model(
                MODEL_ID, pretrained=True, num_classes=0,
            ).eval().requires_grad_(False).to(self.device)

        dataset = TaskImages(self.datasets[split], self.config["class_order"], task)
        expected = 5000 if split == "train" else 1000
        if len(dataset) != expected:
            raise ValueError(f"Unexpected CIFAR-100 shard size for {split} task {task}")
        loader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.workers,
            pin_memory=self.device.type == "cuda", shuffle=False,
        )
        batches = []
        with torch.inference_mode():
            for images, _ in loader:
                embedding = self.model(images.to(self.device, non_blocking=True))
                batches.append(embedding.float().cpu())
        features = torch.cat(batches)
        if features.shape != (expected, FEATURE_DIM) or not torch.isfinite(features).all():
            raise ValueError("Invalid ViT embeddings. Check the backbone and preprocessing.")
        return features, torch.from_numpy(dataset.labels)
