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


def split_class_order(seed=SEED, *, num_classes=NUM_CLASSES):
    """Return the random class order that defines the task split.

    Args:
        seed: Seed for the permutation (default 1993).
        num_classes: Number of classes to permute.

    Returns:
        List of class ids; task t uses entries [10*(t-1), 10*t).
    """
    # RandomState gives the same permutation as np.random.seed + permutation.
    return np.random.RandomState(seed).permutation(num_classes).tolist()


class NCM(nn.Module):
    """Nearest-class-mean classifier with cosine similarity (Phase 2 baseline)."""

    def __init__(self, feature_dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.register_buffer("sums", torch.zeros(num_classes, feature_dim))
        self.register_buffer("counts", torch.zeros(num_classes, dtype=torch.long))

    def update(self, features, labels):
        """Accumulate per-class feature sums and counts.

        Args:
            features: (N, D) embeddings.
            labels: (N,) integer class ids.
        """
        self.sums.index_add_(0, labels, features)
        self.counts += torch.bincount(labels, minlength=self.counts.numel())

    def weights(self, seen_classes):
        """Return the (D, seen_classes) matrix of L2-normalised class means."""
        means = self.sums[:seen_classes] / self.counts[:seen_classes, None]
        return F.normalize(means, dim=1).T

    def logits(self, features, weights):
        """Cosine similarity between normalised features and class means."""
        return F.normalize(features, dim=1) @ weights


class Ridge(nn.Module):
    """Ridge-regression classifier with optional random projection (RanPAC Phase 2).

    Keeps the accumulated Gram matrix G = sum H^T H and correlation C = sum H^T Y,
    where H = ReLU(F W) with a fixed Gaussian W (or H = F if projection_dim is 0).

    Args:
        feature_dim: Backbone feature size D.
        num_classes: Total number of classes.
        projection_dim: Random projection size M (0 disables projection).
        seed: Seed for the Gaussian projection matrix.
    """

    def __init__(self, feature_dim, num_classes=NUM_CLASSES, projection_dim=0, seed=SEED):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "W", torch.randn(feature_dim, projection_dim, generator=generator)
            if projection_dim else None,
        )
        dim = projection_dim or feature_dim
        self.register_buffer("G", torch.zeros(dim, dim))
        self.register_buffer("C", torch.zeros(dim, num_classes))

    def project(self, features):
        """Apply the fixed random projection and ReLU (identity if no projection)."""
        if self.W is None:
            return features
        return (features @ self.W).relu_()

    def update(self, projected, labels):
        """Add a batch to G and C.

        Args:
            projected: (N, M) output of project().
            labels: (N,) integer class ids.
        """
        targets = F.one_hot(labels, num_classes=self.C.shape[1]).to(projected.dtype)
        self.G.addmm_(projected.T, projected)
        self.C.addmm_(projected.T, targets)

    def weights(self, seen_classes, ridge):
        """Solve (G + ridge I) W = C for the first seen_classes columns.

        Returns:
            (M, seen_classes) classifier weights.
        """
        # Promote if a small lambda could be lost when added to the FP32 diagonal.
        resolution = self.G.diagonal().max().item() * torch.finfo(self.G.dtype).eps
        dtype = torch.float64 if ridge <= resolution else self.G.dtype
        system = self.G.to(dtype=dtype, copy=True)
        system.diagonal().add_(ridge)
        return torch.linalg.solve(system, self.C[:, :seen_classes].to(dtype)).to(self.C.dtype)

    def logits(self, features, weights):
        """Project raw features and score them with the given weights."""
        return self.project(features) @ weights


def select_ridge(projected, labels, num_classes, seed, task, grid=RIDGE_GRID):
    """Select lambda on an 80/20 split of the current task's train samples.

    Args:
        projected: (N, M) projected training features of the current task.
        labels: (N,) class ids.
        num_classes: Total number of classes (one-hot width).
        seed: Base seed; the split uses seed + task.
        task: 1-based task index.
        grid: Candidate lambda values.

    Returns:
        The lambda with the lowest validation MSE.
    """
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


def evaluate_seen(classifier, weights, test_shards, device, batch_size=512):
    """Per-shard accuracy (%) of one class-incremental head.

    Each shard groups the metric only; all examples use the same head.

    Args:
        classifier: Object exposing logits(features, weights).
        weights: Classifier weights over all seen classes.
        test_shards: Iterable of (features, labels), one per task.
        device: Device used for scoring.
        batch_size: Scoring batch size.

    Returns:
        List of accuracies in percent, one per shard.
    """
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
