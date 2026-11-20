# Experimental Methodology — RanPAC Reproduction

> Paper: McDonnell et al., *"RanPAC: Random Projections and Pre-trained Models for Continual Learning"*, NeurIPS 2023.

---

## 1. Dataset & Preprocessing

### 1.1 Benchmark: Split CIFAR-100
All reproduction experiments are conducted on **Split CIFAR-100**, the standard benchmark for class-incremental learning derived from CIFAR-100 (60,000 images across 100 classes, original resolution $32 \times 32$).

| Property | Specification |
|---|---|
| Total classes | 100 |
| Total tasks ($T$) | 10 sequential tasks |
| Classes per task | 10 mutually disjoint classes per task |
| Training samples | 500 images per class (5,000 per task, 50,000 total) |
| Test samples | 100 images per class (1,000 per task, 10,000 total) |
| Class order generation | Permutation of classes $0 \dots 99$ via `np.random.RandomState(seed=1993)` |
| Random seed | 1993 (reproducing the exact author split) |

### 1.2 Data Preprocessing & Augmentation Pipeline
Because the backbone expects $224 \times 224$ images, inputs are resized and transformed as follows:

- **Phase 1 Training (Task 1 only, for AdaptFormer):**
  - `RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3/4, 4/3))`
  - `RandomHorizontalFlip(p=0.5)`
  - `ToTensor()`
- **Phase 2 (Feature Extraction & Evaluation across all tasks):**
  - `Resize(224, interpolation=bicubic)`
  - `CenterCrop(224)`
  - `ToTensor()`
  *(Note: Standard channel-wise normalization is omitted, following the official RanPAC CIFAR-100 recipe).*

---

## 2. Evaluation Metrics

Experiments adhere to standard Class-Incremental Learning (CIL) evaluation standards without task identity at inference time.

### 2.1 Final Average Accuracy ($A[T]$)
The primary metric is the overall accuracy over all $K = 100$ classes measured after completing the final task $T = 10$:

$$
A[T] = \frac{1}{K} \sum_{k=1}^{K} \text{Acc}_k^{(T)}
$$

where $\text{Acc}_k^{(T)}$ denotes the top-1 test accuracy on class $k$ after learning task $T$.

### 2.2 Incremental Accuracy Curve ($A[t]$)
After each task $t \in \{1, \dots, T\}$, the model is evaluated on the test sets of **all classes observed up to task $t$** ($10t$ classes):

$$
A[t] = \frac{1}{10t} \sum_{k=1}^{10t} \text{Acc}_k^{(t)}
$$

### 2.3 Forgetting & Backward Transfer Matrix ($R[t, i]$)
To quantify Catastrophic Forgetting, the accuracy matrix $R \in \mathbb{R}^{T \times T}$ is computed, where $R[t, i]$ represents the test accuracy on task $i$'s classes measured immediately after completing task $t$ ($t \ge i$):

- **Average Forgetting ($F$):** Quantifies the average drop from each task's peak performance:
  $$
  F = \frac{1}{T - 1} \sum_{i=1}^{T - 1} \left( \max_{t \in \{i, \dots, T-1\}} R[t, i] - R[T, i] \right)
  $$
- **Backward Transfer ($\text{BWT}$):** Measures whether learning new tasks influences performance on previous tasks (a negative value signifies forgetting):
  $$
  \text{BWT} = \frac{1}{T - 1} \sum_{i=1}^{T - 1} \big(R[T, i] - R[i, i]\big) = -F
  $$

---

## 3. Experimental Protocol

### 3.1 Continual Learning Paradigm
- **Strictly Rehearsal-Free (Zero Exemplars):** No training samples from previous tasks $\mathcal{D}_{<t}$ are stored or replayed.
- **Task-Agnostic Inference:** Task ID is not provided at test time. Predictions are made over the entire label space of all seen classes.
- **Single-Pass Ridge Accumulation:** Features of task $t$ are processed in a single pass to update sufficient statistics; no multi-epoch iterative training or gradient descent is performed during Phase 2.

### 3.2 Pre-trained Backbone Architecture
- **Model:** Vision Transformer **ViT-B/16** (`vit_base_patch16_224.augreg_in21k`), pre-trained on ImageNet-21K with AugReg, loaded via `timm`.
- **Feature Dimension:** Produces embedding vectors $\mathbf{f} \in \mathbb{R}^L$ with $L = 768$.

---

### 3.3 Two-Phase Execution Pipeline

#### Phase 1: First-Session Adaptation (Task 1 Only, Optional)
To bridge the distribution shift between ImageNet-21K and CIFAR-100:
1. **Architecture (AdaptFormer):** A bottleneck adapter branch is inserted parallel to the feed-forward network (FFN) in each of the 12 transformer blocks:
   $$
   \mathbf{x}_{\text{out}} = \mathbf{x} + \text{FFN}(\mathbf{x}) + s \cdot \left( \mathbf{W}_{\text{up}} \, \text{Dropout}(\text{ReLU}(\mathbf{W}_{\text{down}} \mathbf{x} + \mathbf{b}_{\text{down}})) + \mathbf{b}_{\text{up}} \right)
   $$
   where bottleneck dimension $d = 64$, scale factor $s = 0.1$, and dropout rate $p = 0.1$.
2. **Optimization:** Trained on Task 1 for 20 epochs using SGD with Momentum:
   - Learning rate: $\eta = 0.01$ decayed to 0 via Cosine Annealing.
   - Momentum: $0.9$, Weight decay: $5 \times 10^{-4}$, Batch size: $48$.
   - Temporary classification head: Learnable Cosine Similarity head with cross-entropy loss.
3. **Permanent Freezing:** After epoch 20, the temporary head is discarded, and the entire extractor (backbone + adapters) is frozen permanently.

#### Phase 2: Continual Classifier Construction (All Tasks $t = 1 \dots 10$)
1. **Feature Extraction:** $\mathbf{f} = \text{Backbone}(\mathbf{x}) \in \mathbb{R}^L$.
2. **Random Projection with Non-linearity:**
   $$
   \mathbf{h} = \text{ReLU}(\mathbf{W}^\top \mathbf{f}) \in \mathbb{R}^M
   $$
   where $\mathbf{W} \in \mathbb{R}^{L \times M}$ is randomly initialized once using $\mathcal{N}(0, 1)$ with fixed seed and kept frozen.
3. **Sufficient Statistics Accumulation:**
   $$
   \mathbf{G} \leftarrow \mathbf{G} + \sum_{i \in \mathcal{D}_t} \mathbf{h}_i \mathbf{h}_i^\top, \qquad \mathbf{C} \leftarrow \mathbf{C} + \sum_{i \in \mathcal{D}_t} \mathbf{h}_i \mathbf{y}_i^\top
   $$
   where $\mathbf{G} \in \mathbb{R}^{M \times M}$ and $\mathbf{C} \in \mathbb{R}^{M \times K_{\text{total}}}$.
4. **Ridge Regularization Selection ($\lambda$):**
   - Candidate grid: $\lambda \in \{10^{-8}, 10^{-7}, \dots, 10^7, 10^8\}$ (17 values).
   - Evaluated on an 80/20 train/validation split of task $t$'s data.
   - **Dual Formulation Optimization:** When $M > N_{\text{train}}$ (e.g., $M = 10,000, N_{\text{train}} = 4,000$), an eigendecomposition of the kernel Gram matrix $\mathbf{H}_{\text{val}} \mathbf{H}_{\text{train}}^\top (\mathbf{H}_{\text{train}} \mathbf{H}_{\text{train}}^\top + \lambda \mathbf{I})^{-1} \mathbf{Y}_{\text{train}}$ evaluates all 17 candidate $\lambda$ values in a single eigendecomposition pass.
5. **Closed-Form Classifier Head Solution:**
   $$
   \mathbf{W}_o = (\mathbf{G} + \lambda_t \mathbf{I})^{-1} \mathbf{C}_{[:, :K_{\text{seen}}]}
   $$
6. **Inference:**
   $$
   \hat{y} = \arg\max_{k \in \{1, \dots, K_{\text{seen}}\}} (\mathbf{W}_o^\top \mathbf{h})_k
   $$

---

### 3.4 Evaluated Configurations (Ablation Scheme)

To rigorously validate the impact of each theoretical component, five experimental configurations are evaluated:

| # | Configuration | Phase 1 (PETL) | Projection Dim ($M$) | $\lambda$ Selection | Classifier | Purpose |
|:---:|---|:---:|:---:|:---:|:---|---|
| 1 | **NCM Cosine** | ✗ | None ($L=768$) | — | Cosine similarity to class mean | Baseline: prototype-only |
| 2 | **Ridge (no RP)** | ✗ | None ($L=768$) | Task-CV | Linear Ridge regression | Isolates 2nd-order statistics |
| 3 | **RanPAC (Table A5)** | ✗ | $M = 2,500$ | Fixed $\lambda = 100$ | Ridge + RP + ReLU | Evaluates RP with compact $M$ |
| 4 | **RanPAC (Table 1)** | ✗ | $M = 10,000$ | Task-CV | Ridge + RP + ReLU | Full Phase 2 (Table 1 ablation) |
| 5 | **Algorithm 1** | ✓ (AdaptFormer) | $M = 10,000$ | Task-CV | Ridge + RP + ReLU | Complete RanPAC method |

---

### 3.5 Computational Environment & Implementation Specifications
- **Local Workstation:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), CUDA 12.4, Python 3.11, PyTorch 2.6.0.
- **Cloud Environment:** Google Colab (Tesla T4 GPU 15 GB VRAM / Python 3.11).
- **Core Libraries:** `timm 1.0.31` / `1.0.22`, `torchvision 0.21.0`, `wandb 0.31.0`.
- **Reproducibility Guarantee:** Random seeds for class ordering (`1993`), projection initialization, and validation splits are strictly seeded across all runs.
