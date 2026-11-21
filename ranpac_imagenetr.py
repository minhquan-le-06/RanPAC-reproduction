"""ImageNet-R split and SSF for the group's Algorithm 1 runner."""

import hashlib
from copy import copy
from pathlib import Path

import torch
from torch import nn


NUM_CLASSES = 200
CLASSES_PER_TASK = 20
MODEL_ID = "vit_base_patch16_224.augreg_in21k_ft_in1k"
CHECKPOINT_URL = (
    "https://storage.googleapis.com/vit_models/augreg/"
    "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0"
    "--imagenet2012-steps_20k-lr_0.01-res_224.npz"
)


class ImageNetR:
    def __init__(self, root):
        from torchvision.datasets import ImageFolder

        root = Path(root)
        self.splits = {split: ImageFolder(root / split) for split in ("train", "test")}
        train, test = self.splits["train"], self.splits["test"]
        if len(train.classes) != NUM_CLASSES or train.class_to_idx != test.class_to_idx:
            raise ValueError("ImageNet-R train/ and test/ must contain the same 200 classes.")
        digest = hashlib.sha256()
        for split in self.splits.values():
            for filename, label in split.samples:
                path = Path(filename)
                relative = path.relative_to(root).as_posix()
                digest.update(f"{label}\t{relative}\t{path.stat().st_size}\n".encode())
        self.manifest_sha256 = digest.hexdigest()

    def image_folder(self, split, transform):
        dataset = copy(self.splits[split])
        dataset.transform = transform
        return dataset


class ScaleShift(nn.Module):
    """Train one affine scale and shift per output channel."""

    def __init__(self, width):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(width))
        self.shift = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.scale, mean=1.0, std=0.02)
        nn.init.normal_(self.shift, std=0.02)

    def forward(self, features):
        return features * self.scale.to(features.dtype) + self.shift.to(features.dtype)


def attach_ssf(backbone):
    """Insert SSF after patch embedding, the six block projections/norms, and final norm."""
    backbone.requires_grad_(False)
    width = backbone.embed_dim
    parameters = nn.ModuleList()

    def adapt(module, width):
        affine = ScaleShift(width)
        parameters.append(affine)
        return nn.Sequential(module, affine)

    backbone.patch_embed.norm = adapt(backbone.patch_embed.norm, width)
    for block in backbone.blocks:
        block.norm1 = adapt(block.norm1, width)
        block.attn.qkv = adapt(block.attn.qkv, 3 * width)
        block.attn.proj = adapt(block.attn.proj, width)
        block.norm2 = adapt(block.norm2, width)
        block.mlp.fc1 = adapt(block.mlp.fc1, block.mlp.fc1.out_features)
        block.mlp.fc2 = adapt(block.mlp.fc2, width)
    backbone.norm = adapt(backbone.norm, width)
    return parameters


def build_ssf_backbone(device):
    import timm

    timm.layers.set_fused_attn(False)
    backbone = timm.create_model(
        MODEL_ID, pretrained=True, num_classes=0,
        pretrained_cfg_overlay={"hf_hub_id": None, "url": CHECKPOINT_URL},
    )
    parameters = attach_ssf(backbone)
    return backbone.to(device), parameters
