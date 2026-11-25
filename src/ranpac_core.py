"""Split CIFAR-100 protocol and the three Phase 2 classifiers."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


NUM_CLASSES = 100
CLASSES_PER_TASK = 10
NUM_TASKS = NUM_CLASSES // CLASSES_PER_TASK
SEED = 1993
RIDGE_GRID = tuple(10.0**power for power in range(-8, 9))
ACTIVATIONS = ("relu", "linear", "gelu", "tanh", "leaky_relu", "abs")


def apply_activation(projected, activation, *, inplace=False):
    """Apply elementwise nonlinearity after random projection."""
    if activation == "relu":
        return projected.relu_() if inplace else F.relu(projected)
    if activation == "linear":
        return projected
    if activation == "gelu":
        return F.gelu(projected)
    if activation == "tanh":
        return torch.tanh(projected)
    if activation == "leaky_relu":
        return F.leaky_relu(projected, negative_slope=0.1, inplace=inplace)
    if activation == "abs":
        return projected.abs_() if inplace else projected.abs()
    raise ValueError(f"Unsupported activation {activation!r}. Choose from {ACTIVATIONS}.")


def split_class_order(seed=SEED, *, num_classes=NUM_CLASSES):
    # RandomState reproduces the authors' np.random.seed + permutation.
    return np.random.RandomState(seed).permutation(num_classes).tolist()


class NCM(nn.Module):
    def __init__(self, feature_dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.register_buffer("sums", torch.zeros(num_classes, feature_dim))
        self.register_buffer("counts", torch.zeros(num_classes, dtype=torch.long))

    def update(self, features, labels):
        self.sums.index_add_(0, labels, features)
        self.counts += torch.bincount(labels, minlength=self.counts.numel())

    def weights(self, seen_classes):
        means = self.sums[:seen_classes] / self.counts[:seen_classes, None]
        return F.normalize(means, dim=1).T

    def logits(self, features, weights):
        return F.normalize(features, dim=1) @ weights


class Ridge(nn.Module):
    def __init__(self, feature_dim, num_classes=NUM_CLASSES, projection_dim=0, seed=SEED,
                 use_relu=True, *, activation=None):
        super().__init__()
        if activation is None:
            activation = "relu" if use_relu else "linear"
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation {activation!r}. Choose from {ACTIVATIONS}.")
        self.activation = activation
        self.use_relu = activation == "relu"
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "W", torch.randn(feature_dim, projection_dim, generator=generator)
            if projection_dim else None,
        )
        dim = projection_dim or feature_dim
        self.register_buffer("G", torch.zeros(dim, dim))
        self.register_buffer("C", torch.zeros(dim, num_classes))

    def project(self, features):
        if self.W is None:
            return features
        projected = features @ self.W
        return apply_activation(projected, self.activation, inplace=True)

    def update(self, projected, labels):
        targets = F.one_hot(labels, num_classes=self.C.shape[1]).to(projected.dtype)
        self.G.addmm_(projected.T, projected)
        self.C.addmm_(projected.T, targets)

    def weights(self, seen_classes, ridge):
        # Promote if a small lambda could be lost when added to the FP32 diagonal.
        resolution = self.G.diagonal().max().item() * torch.finfo(self.G.dtype).eps
        dtype = torch.float64 if ridge <= resolution else self.G.dtype
        system = self.G.to(dtype=dtype, copy=True)
        system.diagonal().add_(ridge)
        return torch.linalg.solve(system, self.C[:, :seen_classes].to(dtype)).to(self.C.dtype)

    def logits(self, features, weights):
        return self.project(features) @ weights


def select_ridge(projected, labels, num_classes, seed, task, grid=RIDGE_GRID):
    """Select lambda on an 80/20 split of the current task's train samples."""
    generator = torch.Generator().manual_seed(seed + task)
    indices = torch.randperm(len(labels), generator=generator).to(labels.device)
    split = int(0.8 * len(labels))
    features = projected[indices].to(torch.float64)
    targets = F.one_hot(labels[indices], num_classes=num_classes).to(torch.float64)
    train, val = features[:split], features[split:]
    train_y, val_y = targets[:split], targets[split:]

    if train.shape[1] > len(train):
        # Dual ridge: val @ train.T @ (train @ train.T + lambda I)^-1 @ train_y.
        # Reuse one eigendecomposition for all lambdas in the large-RP sweep.
        values, vectors = torch.linalg.eigh(train @ train.T)
        values.clamp_min_(0)
        left = (val @ train.T) @ vectors
        right = vectors.T @ train_y
        losses = [F.mse_loss(left @ (right / (values[:, None] + ridge)), val_y).item()
                  for ridge in grid]
    else:
        gram = train.T @ train
        cross = train.T @ train_y
        identity = torch.eye(len(gram), device=gram.device, dtype=gram.dtype)
        losses = [F.mse_loss(val @ torch.linalg.solve(gram + ridge * identity, cross), val_y).item()
                  for ridge in grid]
    return min(zip(losses, grid))[1]


def gram_spectrum_stats(gram):
    """Compute numerical rank diagnostics from an accumulated symmetric Gram matrix G."""
    m = gram.shape[0]
    eps = torch.finfo(gram.dtype).eps
    ev = torch.linalg.eigvalsh(gram.to(torch.float64))
    ev_pos = ev.clamp_min(0.0)
    lam_max = float(ev_pos[-1].item()) if m else 0.0
    lam_second = float(ev_pos[-2].item()) if m > 1 else lam_max
    tol_standard = lam_max * m * eps
    rank_standard = int((ev > tol_standard).sum().item())
    if m > 1 and lam_max > 0:
        tol_deflated = lam_second * m * eps
        rank_deflated_dc = 1 + int((ev[:-1] > tol_deflated).sum().item())
    else:
        tol_deflated = tol_standard
        rank_deflated_dc = rank_standard
    total = float(ev_pos.sum().item())
    if total > 0:
        probs = ev_pos[ev_pos > 0] / total
        effective_rank = float(torch.exp(-(probs * probs.log()).sum()).item())
    else:
        effective_rank = 0.0
    return {
        "M": m,
        "g_dtype": str(gram.dtype),
        "lambda_max": lam_max,
        "lambda_second": lam_second,
        "top1_to_top2_ratio": (lam_max / lam_second) if lam_second > 0 else float("inf"),
        "tol_standard": tol_standard,
        "rank_standard": rank_standard,
        "tol_deflated_dc": tol_deflated,
        "rank_deflated_dc": rank_deflated_dc,
        "effective_rank": effective_rank,
    }


def evaluate_seen(classifier, weights, test_shards, device, batch_size=512):
    """Each shard groups the metric only. All examples use the same CIL head."""
    accuracies = []
    for features, labels in test_shards:
        correct = 0
        for start in range(0, len(labels), batch_size):
            end = start + batch_size
            batch = features[start:end].to(device)
            truth = labels[start:end].to(device)
            correct += (classifier.logits(batch, weights).argmax(1) == truth).sum().item()
        accuracies.append(100.0 * correct / len(labels))
    return accuracies
