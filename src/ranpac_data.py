"""Dataset loading and shared ViT feature shards for Colab."""

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

from ranpac_core import CLASSES_PER_TASK


MODEL_ID = "vit_base_patch16_224.augreg_in21k"
CHECKPOINT_URL = (
    "https://storage.googleapis.com/vit_models/augreg/"
    "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz"
)
FEATURE_DIM = 768
CIFAR100_HF_ID = "uoft-cs/cifar100"


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


class HFImages(Dataset):
    """Image-classification view of a Hugging Face dataset.

    Args:
        hf_dataset: A `datasets.Dataset` with `img` and `fine_label` columns.
        transform: Transform applied to each PIL image.

    Exposes `targets` (list of labels) so TaskImages can filter by class.
    """

    def __init__(self, hf_dataset, transform):
        self.data = hf_dataset
        self.transform = transform
        self.targets = list(hf_dataset["fine_label"])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        row = self.data[int(index)]
        return self.transform(row["img"].convert("RGB")), int(row["fine_label"])


def load_cifar100(split, transform, data_root):
    """Load CIFAR-100 from the Hugging Face Hub at run time (no manual download).

    Args:
        split: "train" (50,000 images) or "test" (10,000 images).
        transform: Transform applied to each image.
        data_root: Directory used as the Hugging Face cache.

    Returns:
        HFImages dataset whose class ids match the original CIFAR-100 ordering.
    """
    from datasets import load_dataset

    return HFImages(load_dataset(CIFAR100_HF_ID, split=split, cache_dir=str(data_root)), transform)


class TaskImages(Dataset):
    def __init__(self, dataset, class_order, task, classes_per_task=CLASSES_PER_TASK):
        num_classes = len(class_order)
        old_to_new = np.empty(num_classes, dtype=np.int64)
        old_to_new[class_order] = np.arange(num_classes)
        labels = old_to_new[np.asarray(dataset.targets)]
        stop = task * classes_per_task
        self.indices = np.flatnonzero((labels >= stop - classes_per_task) & (labels < stop))
        self.labels = labels[self.indices]
        self.dataset = dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.dataset[int(self.indices[index])]
        return image, int(self.labels[index])


class FeatureCache:
    def __init__(self, root, data_root, config, device, batch_size, workers, *, image_folder=None):
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
        self.root = Path(root) / key
        self.data_root = data_root
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.workers = workers
        self.image_folder = image_folder
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
        from torchvision import transforms

        if split not in self.datasets:
            transform = transforms.Compose([
                transforms.Resize(256 if self.image_folder else 224,
                                  interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
            if self.image_folder is None:
                self.datasets[split] = load_cifar100(split, transform, self.data_root)
            else:
                self.datasets[split] = self.image_folder(split, transform)

        if self.model is None:
            if self.image_folder is not None:
                raise ValueError("Set the adapted backbone before extracting ImageNet-R features.")
            timm.layers.set_fused_attn(False)
            # Use the same AugReg NPZ as timm 0.6.12, even if HF Hub is installed.
            self.model = timm.create_model(
                MODEL_ID, pretrained=True, num_classes=0,
                pretrained_cfg_overlay={"hf_hub_id": None, "url": CHECKPOINT_URL},
            ).eval().requires_grad_(False).to(self.device)

        dataset = TaskImages(self.datasets[split], self.config["class_order"], task,
                             self.config["classes_per_task"])
        if self.image_folder is None and len(dataset) != (5000 if split == "train" else 1000):
            raise ValueError(f"Unexpected CIFAR-100 shard size for {split} task {task}")
        if not len(dataset):
            raise ValueError(f"No samples in {split} task {task}")
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
        if features.shape != (len(dataset), FEATURE_DIM) or not torch.isfinite(features).all():
            raise ValueError("Invalid ViT embeddings. Check the backbone and preprocessing.")
        return features, torch.from_numpy(dataset.labels)
