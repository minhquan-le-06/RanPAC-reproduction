# Experimental Methodology

> Paper: McDonnell et al., *"RanPAC: Random Projections and Pre-trained Models for Continual Learning"*, NeurIPS 2023.

---

## 1. Dataset

We evaluate on **Split CIFAR-100**, a standard benchmark for Class-Incremental Learning (CIL). The dataset is derived from CIFAR-100 (100 classes, 60,000 images at 32×32 pixels) by partitioning it into a sequence of tasks.

| Property | Value |
|---|---|
| Total classes | 100 |
| Number of tasks | 10 |
| Classes per task | 10 (disjoint) |
| Training images per class | 500 |
| Test images per class | 100 |
| Class order | Random permutation, seed 1993 |

The 100 classes are permuted with seed 1993 (matching the original authors' split) and assigned sequentially — Task 1 receives classes 0–9, Task 2 receives classes 10–19, and so on. Task identity is **not** provided at inference time, requiring the model to classify among all classes seen so far.

Since the backbone expects 224×224 inputs, images are resized from 32×32:

- **Phase 1 training** (Task 1 only, with AdaptFormer): `RandomResizedCrop(224)` → `RandomHorizontalFlip` → `ToTensor`.
- **Phase 2 feature extraction** (all tasks): `Resize(224, bicubic)` → `CenterCrop(224)` → `ToTensor`. Channel-wise normalization is omitted following the official RanPAC CIFAR-100 configuration.

---

## 2. Metrics

We follow standard CIL evaluation, measuring accuracy over all classes seen so far — without access to task identity at test time.

**Final average accuracy $A[T]$** is the primary reported metric, computed after all 10 tasks:

$$A[T] = \frac{1}{K}\sum_{k=1}^{K} \mathrm{Acc}_k^{(T)}$$

where $K = 100$ and $\mathrm{Acc}_k^{(T)}$ is the top-1 test accuracy on class $k$ after the final task.

**Incremental accuracy $A[t]$** tracks the average accuracy over all $10t$ classes seen after task $t$:

$$A[t] = \frac{1}{10t}\sum_{k=1}^{10t} \mathrm{Acc}_k^{(t)}$$

This produces an accuracy curve over 10 tasks that reveals how the model balances learning new classes against retaining old ones.

**Task accuracy matrix $R[t, i]$** records accuracy on task $i$'s test set measured immediately after completing task $t$ ($t \geq i$). From this matrix we derive two forgetting metrics:

$$\text{Average Forgetting} = \frac{1}{T-1}\sum_{i=1}^{T-1}\bigl(R[i,i] - R[T,i]\bigr)$$

$$\text{BWT} = \frac{1}{T-1}\sum_{i=1}^{T-1}\bigl(R[T,i] - R[i,i]\bigr) = -\text{Forgetting}$$

A BWT of 0 indicates no forgetting; a negative BWT indicates that old task accuracy degraded after new tasks were learned.

---

## 3. Experimental Protocol

### 3.1 Continual Learning Setting

Experiments follow the **rehearsal-free class-incremental** setting:
- No samples from previous tasks are stored or replayed.
- The model is evaluated over all classes seen so far, without being told which task a test image belongs to.
- After each task, a single closed-form solve updates the classifier — no gradient-based training is performed on past data.

### 3.2 Configurations

Five configurations are evaluated, forming a progression from baseline to full method. Each adds one component to the previous, isolating its contribution:

| # | Configuration | Phase 1 | Projection $M$ | $\lambda$ selection |
|:---:|---|:---:|:---:|:---:|
| 1 | NCM Cosine | ✗ | — | — |
| 2 | Ridge (no RP) | ✗ | — | Task CV |
| 3 | RanPAC (Table A5) | ✗ | 2,500 | Fixed 100 |
| 4 | RanPAC (Table 1) | ✗ | 10,000 | Task CV |
| 5 | Algorithm 1 (full) | ✓ AdaptFormer | 10,000 | Task CV |

Configurations 1–4 use a frozen ViT-B/16 from the start. Configuration 5 adds a Phase 1 step: an AdaptFormer adapter (bottleneck 64, scale 0.1) is trained on Task 1 for 20 epochs using SGD with cosine annealing (lr 0.01, momentum 0.9, weight decay $5\times10^{-4}$), then frozen before Phase 2 begins.

For configurations using ridge regression, the regularization coefficient $\lambda$ is selected per task from the grid $\{10^{-8}, 10^{-7}, \ldots, 10^8\}$ on an 80/20 split of the current task's training data.

### 3.3 Implementation and Environment

| Item | Detail |
|---|---|
| Backbone | ViT-B/16, ImageNet-21K pre-trained (`vit_base_patch16_224.augreg_in21k`) |
| Projection matrix | Gaussian $\mathcal{N}(0,1)$, fixed seed 1993, frozen throughout |
| Framework | PyTorch 2.6.0, timm 1.0.31 |
| Experiment tracking | Weights & Biases (`pnp123/ML` on Forge CoreWeave) |
| Local GPU | RTX 4060 Laptop 8 GB (config 3) |
| Cloud GPU | Google Colab T4 15 GB (configs 1, 2, 4, 5) |
| Random seed | 1993 for class order, projection matrix, and $\lambda$ validation splits |
