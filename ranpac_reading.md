# RanPAC: Random Projections and Pre-trained Models for Continual Learning

Markers: [VERIFY] = not covered in the team's reading notes; check against the paper before use. [TODO] = still to be completed by the team.

Structure follows the course reading guide: Context and motivation → Contributions → Method (notation, pipeline, components, theoretical properties, complexity). Section 4 covers the evaluation methodology required in the deck.

## 1. Context and Motivation

### 1.1 The problem: continual learning

Continual learning (CL) trains a model on a sequence of tasks $\mathcal{D}_1, \mathcal{D}_2, \dots, \mathcal{D}_T$ whose data distribution changes over time. After each task, the model must still perform well on every task seen so far. RanPAC [1] targets the setting without rehearsal, in which samples from earlier tasks are neither stored nor replayed. Two variants are studied:

- **Class-incremental learning (CIL):** each task introduces new classes, and no task identity is available at test time.
- **Domain-incremental learning (DIL):** the label set stays fixed while the input distribution shifts between tasks.

Naive fine-tuning fails in this setting for two reasons. First, gradient updates on new data overwrite the parameters that encoded earlier classes. Second, because tasks are trained in isolation, the model never learns to separate old classes from new ones. Together these effects are known as catastrophic forgetting. Replaying stored samples mitigates forgetting, but storing data is often ruled out by privacy or cost.

### 1.2 Why it matters: two concrete cases

**Medical imaging.** A diagnostic model first learns to detect one disease and is later extended to new diseases, imaging devices or hospitals. Retraining on old and new data together is computationally costly and runs into storage constraints and patient privacy concerns, so the scans used for earlier tasks often cannot be kept. Yet performance on the earlier diseases must not degrade. The review by Kumari et al. [10] identifies these constraints as the main motivation for continual learning in medical image analysis.

**Agricultural advisory app.** The 2Nông app by Petrovietnam Ca Mau Fertilizer (PVCFC) added an AI feature in December 2024 that diagnoses pests, diseases and nutrient deficiencies from a photo of the crop [11], [12]. Reported coverage has grown over time, from coffee and pepper to six crop groups including rice, durian and passion fruit, with more than 60 classes of pests, diseases and nutrient deficiencies. Each extension adds new classes from photos that farmers upload sporadically. Retraining on the full image history at every update is impractical, and plain fine-tuning on the new crops would erode accuracy on the crops already supported. Coverage figures differ between sources and are reported by the company [VERIFY].

Both cases require the same three properties: new classes or domains can be added cheaply, old data is not needed, and performance on what has already been learned is not degraded.

### 1.3 The opportunity: pre-trained foundation models

Large pre-trained models such as ViT, CLIP and ResNet already produce general features. This changes the question. Instead of asking how to protect a representation while it is being learned, one can ask how to build a classifier on top of a frozen, strong representation that keeps improving as tasks arrive. The predecessor approaches below differ mainly in how much they still modify the backbone.

### 1.4 Predecessor approaches

Table 1 is redrawn from Table A1 of the paper. A check mark means the authors credit the family with the property, and an empty cell means they do not. The "?" is the authors' own mark: they do not commit to whether fine-tuning approaches transfer to CNN backbones, since the methods they cite are reported on ViTs [VERIFY the meaning in the paper's caption or text]. The ratings are the authors' assessment, not ours.

| Criterion | Prompting | Fine-tuning | CP | CP + RP (RanPAC) |
|---|---|---|---|---|
| No rehearsal buffer | ✓ | ✓ | ✓ | ✓ |
| Pre-trained model frozen | ✓ | | ✓ | ✓ |
| Transformers and CNNs | | ? | ✓ | ✓ |
| Simplicity | | | ✓ | ✓ |
| Parameter-efficient | ✓ | | ✓ | ✓ |
| Theoretical support | | | | ✓ |
| SOTA performance | | | | ✓ |

#### 1.4.1 Prompting (L2P, DualPrompt, CODA-Prompt)

**Core idea.** These methods keep the pre-trained transformer frozen and insert small learnable prompt vectors into the token sequence or into intermediate layers. L2P [2] selects prompts from a pool by matching a query feature against learned keys. DualPrompt [3] separates prompts shared across tasks from prompts specific to each task. CODA-Prompt [4] composes prompts with an attention mechanism.

**Remaining limitation.** The prompts and keys are trained by gradient descent on each new task. The overwriting mechanism behind forgetting therefore still applies, only to fewer parameters. Prompting also depends on the token structure of transformers.

**Practical consequence.** Forgetting is reduced but not eliminated, and the approach cannot be used with CNN backbones.

#### 1.4.2 Selective fine-tuning (e.g., SLCA)

**Core idea.** The backbone is updated in a controlled way, with a very small learning rate or only in selected blocks. SLCA [5] combines a slow backbone learning rate with classifier alignment and reports better results than prompting.

**Remaining limitation.** Every task requires backpropagation through the whole network. To realign its classifier, SLCA also stores a mean and a covariance matrix for each class.

**Practical consequence.** Compute and memory costs are high, and memory grows with the number of classes. Because the backbone still moves, these methods usually need additional CL techniques to stay stable, which makes an already expensive pipeline more complex.

#### 1.4.3 Class prototypes on frozen features (NCM, ADaM)

**Core idea.** Each class is represented by a prototype, the mean of its feature vectors. Prototypes are accumulated across tasks, and a sample is assigned to the class whose prototype has the highest cosine similarity (nearest class mean, NCM). Nothing is trained after feature extraction, so forgetting is impossible by construction. ADaM [6] adds parameter-efficient transfer learning (PETL) on the first task before freezing the backbone.

**Remaining limitation.** A mean ignores the spread and correlation of a class distribution, so classifying by distance to the mean implicitly assumes isotropic classes with equal variance. Pre-trained features violate this assumption: their dimensions are strongly correlated, so the prototypes of different classes are themselves highly similar. There is also a distribution gap between the pre-training data and the new data.

**Practical consequence.** Accuracy reaches a ceiling, and visually similar classes are frequently confused.

#### 1.4.4 Prototypes with second-order statistics (First Session Adaptation)

**Core idea.** First Session Adaptation (FSA) [7] adapts the backbone on the first task and then freezes it. From the second task onwards, it accumulates a covariance matrix across tasks and multiplies features by its inverse before comparing them with prototypes, which amounts to incremental LDA. Removing the components shared across dimensions decorrelates the prototypes and drops the assumption of equal variance, which clearly improves on plain NCM.

**Remaining limitation.** The classifier is still linear in the backbone's original feature space of dimension $L$, and no interactions between features are created.

**Practical consequence.** Class separability is capped by what is linearly separable in the backbone space.

### 1.5 The gap RanPAC fills

Methods based on class prototypes need no rehearsal, cannot forget, and are cheap, but their accuracy is limited by a linear classifier on the raw backbone features. RanPAC keeps the training-free accumulation of these methods. It first projects the features into a space of much higher dimension using a frozen random matrix followed by a nonlinearity. This creates interactions between features and makes classes easier to separate. No parameters are trained in this step, so it introduces no forgetting.

## 2. Contributions

### 2.1 As stated by the authors

1. **An improved class-prototype strategy.** Adding a random projection (RP) with a nonlinear activation improves the linear separability of features. The paper also analyses why second-order statistics are needed to decorrelate class prototypes, following [7].
2. **RP combined with PETL on the first task.** This combination brings accuracy close to joint full fine-tuning on several datasets. With a ViT-B/16 backbone, RanPAC reported the best results at the time of publication among CL methods without rehearsal on the CIL and DIL benchmarks studied.
3. **Generality.** RanPAC works with features from ViT, ResNet and CLIP, and applies to CIL, DIL and task-agnostic CL.

### 2.2 By kind, with novelty relative to predecessors

**Theoretical.** The paper analyses how RP and the nonlinearity affect the feature representation. It also shows that a classifier head built on the Gram matrix is a ridge regression solution and that it decorrelates the class prototypes (Sec. 4.1–4.3, App. B). Earlier RP work [8], [9] had already studied random nonlinear features outside CL, and FSA [7] had already used second-order statistics via incremental LDA. RanPAC's contribution is to bring this analysis to a CL method based on class prototypes with a pre-trained backbone. Neither the ridge regression formula nor the idea of decorrelation is new.

**Technical.** RanPAC is a pipeline with two phases: optional PETL on the first task, a frozen extractor, a nonlinear RP that expands the dimension, accumulation of the statistics $\mathbf{G}$ and $\mathbf{C}$, and a classifier head computed in closed form. After the first phase, there is neither SGD nor replay. The novelty lies in how the components are combined:

| Predecessor | What it already had | What RanPAC adds |
|---|---|---|
| NCM / ADaM [6] | Class prototypes; PETL on the first task | Nonlinear RP and a head built on the Gram matrix instead of cosine similarity |
| FSA [7] | Adaptation on the first session; incremental LDA | Nonlinear RP; ridge regression form of the head |
| RP literature [8], [9] | Random nonlinear features; least squares readout | Use within CL with a pre-trained backbone and accumulated statistics |

**Empirical.** On seven CIL benchmarks (Table 1 of the paper), RanPAC reduces the relative error rate by about 20–62% compared with ADaM. On several datasets it approaches joint full fine-tuning (Table 2 of the paper). Ablations isolate the contributions of RP, PETL and second-order statistics. Experiments across backbones and across the CIL, DIL and task-agnostic settings establish the method's scope. The task-agnostic experiment (App. F.4) uses Phase 2 only.

## 3. Method

### 3.1 Notation

| Symbol | Meaning | Dimension |
|---|---|---|
| $T,\ t$ | Number of tasks; task index | — |
| $\mathcal{D}_t$ | Training data of task $t$ | — |
| $K$ | Number of classes seen so far | — |
| $\mathbf{f}$ | Backbone feature of an input | $\mathbb{R}^{L}$ |
| $\mathbf{W}$ | Frozen random projection matrix | $\mathbb{R}^{L \times M},\ M > L$ |
| $\phi(\cdot)$ | Nonlinear activation applied to each element | — |
| $\mathbf{h} = \phi(\mathbf{W}^\top \mathbf{f})$ | Expanded feature | $\mathbb{R}^{M}$ |
| $\mathbf{y}$ | Label vector: 1 at the true class, 0 elsewhere | $\lbrace 0,1 \rbrace^{K}$ |
| $\mathbf{H}$ | All expanded features seen so far, stacked as rows | $\mathbb{R}^{N \times M}$ |
| $\mathbf{Y}$ | All labels seen so far, stacked as rows | $\mathbb{R}^{N \times K}$ |
| $\mathbf{G}$ | Gram matrix, Eq. (1) | $\mathbb{R}^{M \times M}$ |
| $\mathbf{C}$ | Sum of features per class, Eq. (2) | $\mathbb{R}^{M \times K}$ |
| $\lambda$ | Ridge regularisation coefficient | scalar |
| $\mathbf{W}_{o}$ | Classifier head, Eq. (3) | $\mathbb{R}^{M \times K}$ |
| $\mathbf{s}$ | Class scores, Eq. (4) | $\mathbb{R}^{K}$ |

$$
\mathbf{G} = \mathbf{H}^\top\mathbf{H} = \sum_{i=1}^{N} \mathbf{h}_i\mathbf{h}_i^\top \tag{1}
$$

$$
\mathbf{C} = \mathbf{H}^\top\mathbf{Y} = \sum_{i=1}^{N} \mathbf{h}_i\mathbf{y}_i^\top \tag{2}
$$

$$
\mathbf{W}_o = (\mathbf{G} + \lambda\mathbf{I})^{-1}\mathbf{C} \tag{3}
$$

$$
\mathbf{s} = \mathbf{W}_o^\top \mathbf{h}, \qquad \hat{y} = \arg\max_k s_k \tag{4}
$$

Column $k$ of $\mathbf{C}$ is the sum of the expanded features of class $k$, i.e. an unnormalised class prototype. $\mathbf{G}$ is uncentered: it is built from raw inner products, not from deviations around a mean, so it is not a covariance matrix. [VERIFY] In the paper, $\mathbf{W}$ has i.i.d. Gaussian entries, $\phi$ is ReLU, and $M \approx 10{,}000$ by default.

### 3.2 Formal problem

Tasks $\mathcal{D}_1, \dots, \mathcal{D}_T$ arrive sequentially, and data from $\mathcal{D}_{<t}$ cannot be accessed while learning $\mathcal{D}_t$. After each task, the model must classify inputs from all $K$ classes seen so far. RanPAC's classifier head solves a ridge regression problem over all data seen so far:

$$
\mathbf{W}_o^{*} = \arg\min_{\mathbf{W}_o}\ \lVert \mathbf{Y} - \mathbf{H}\mathbf{W}_o \rVert_F^2 + \lambda \lVert \mathbf{W}_o \rVert_F^2
$$

The solution depends on the data only through $\mathbf{G} = \mathbf{H}^\top\mathbf{H}$ and $\mathbf{C} = \mathbf{H}^\top\mathbf{Y}$. Both are sums over samples, so the problem can be solved exactly even though old samples are never revisited (Section 3.5, Property 1). Performance is measured by the final average accuracy over all classes seen [VERIFY].

### 3.3 Overall pipeline

RanPAC runs in two phases.

**Phase 1: extractor preparation (task 1 only, optional).** Starting from a pre-trained backbone, PETL parameters are trained on the first task and then frozen. From this point on, the way features are extracted never changes.

**Phase 2: classifier construction (every task, including task 1).**

1. **Feature extraction.** Each input of the current task passes through the frozen backbone to produce $\mathbf{f}$.
2. **Random projection.** The feature is expanded and passed through the nonlinearity: $\mathbf{h} = \phi(\mathbf{W}^\top\mathbf{f})$, with $M > L$. $\mathbf{W}$ is generated once and reused for all tasks.
3. **Statistics update.** $\mathbf{G} \leftarrow \mathbf{G} + \mathbf{h}\mathbf{h}^\top$ and $\mathbf{C} \leftarrow \mathbf{C} + \mathbf{h}\mathbf{y}^\top$. Statistics from earlier tasks are kept. Phase 2 also processes task 1, even if task 1 was used to train PETL.
4. **Head computation.** After each task, $\mathbf{W}_o = (\mathbf{G} + \lambda\mathbf{I})^{-1}\mathbf{C}$ is computed from the accumulated statistics. The head covers every class seen so far without accessing old samples.
5. **Prediction.** A test input goes through the same frozen path to produce $\mathbf{h}_{\text{new}}$. The scores are $\mathbf{s} = \mathbf{W}_o^\top\mathbf{h}_{\text{new}}$, and the prediction is $\hat{y} = \arg\max_k s_k$. Nothing is updated at inference time.

[Figure: overall pipeline, Overall_pipeline.drawio]

### 3.4 Components

#### 3.4.1 PETL on the first task (optional)

- **What it does.** A small set of adapter parameters is trained on the first task and then frozen. Possible choices are AdaptFormer, SSF and VPT [VERIFY].
- **Why it is needed.** Pre-trained features fit some downstream datasets poorly. ImageNet-R and ImageNet-A, for example, differ strongly from natural ImageNet photos. A short adaptation on the first task closes part of this gap at low cost, because only a small fraction of parameters is trained.
- **How it works.** Standard SGD on $\mathcal{D}_1$ with a temporary linear head; the original backbone weights are untouched.

Adaptation is restricted to the first task on purpose. If the extractor kept changing, the information about old classes held in $\mathbf{G}$ and $\mathbf{C}$ would no longer match newly extracted features, and forgetting would return. PETL is optional, and its benefit depends on the dataset.

#### 3.4.2 Random projection with a nonlinearity

- **What it does.** It computes $\mathbf{h} = \phi(\mathbf{W}^\top\mathbf{f})$, with $\mathbf{W} \in \mathbb{R}^{L \times M}$ random, frozen and $M \gg L$.
- **Why it is needed.** A linear head on $\mathbf{f}$ can only separate what is linearly separable in the backbone space, which is the ceiling identified in Section 1.4.4. The projection creates nonlinear interactions between features, so classes become more linearly separable in the expanded space.
- **How it works.** $\mathbf{W}$ is sampled once and never trained, so it introduces no forgetting.

A linear classifier in this expanded space can realise decision boundaries that are nonlinear in the original space, in line with Cover's theorem [8] and random feature methods [9]. The nonlinearity is essential: without it, $\mathbf{h} = \mathbf{W}^\top\mathbf{f}$ is a linear function of $\mathbf{f}$ and nothing is gained (Section 3.5, Property 2).

#### 3.4.3 Accumulated statistics $\mathbf{G}$ and $\mathbf{C}$

- **What it does.** For every training sample, $\mathbf{G} \leftarrow \mathbf{G} + \mathbf{h}\mathbf{h}^\top$ and $\mathbf{C} \leftarrow \mathbf{C} + \mathbf{h}\mathbf{y}^\top$.
- **Why it is needed.** These two sums are all the classifier head needs, so samples can be discarded once processed. This is why the method needs no rehearsal.
- **How it works.** The update runs sample by sample and task by task. Storage depends only on $M$ and $K$, not on the number of samples $N$.

$\mathbf{C}$ holds the class prototypes up to a scale, while $\mathbf{G}$ captures second-order structure: how the dimensions of $\mathbf{h}$ vary together over all data seen so far.

#### 3.4.4 Decorrelated classifier head in closed form

- **What it does.** It computes $\mathbf{W}_o = (\mathbf{G} + \lambda\mathbf{I})^{-1}\mathbf{C}$ and predicts $\hat{y} = \arg\max_k (\mathbf{W}_o^\top\mathbf{h})_k$.
- **Why it is needed.** Using $\mathbf{C}$ alone as the head would be NCM with unnormalised prototypes. Multiplying by the inverse Gram matrix removes components shared across classes and keeps what is specific to each class, i.e. it decorrelates the prototypes.
- **How it works.** After each task, one $M \times M$ linear system is solved. The term $\lambda\mathbf{I}$ keeps the matrix well conditioned and makes the inversion stable, which matters because $M$ is large. $\lambda$ is chosen per task from a set of candidate values using a validation split of the current training data [VERIFY].

### 3.5 Theoretical properties

The paper does not state formal theorems. Its theoretical support consists of the arguments below (Sec. 4, App. B) [VERIFY]. For each, we state what it promises and where its limits are.

**Property 1: the head equals joint ridge regression, regardless of task order.** Setting the gradient of the ridge objective to zero gives

$$
(\mathbf{H}^\top\mathbf{H} + \lambda\mathbf{I})\,\mathbf{W}_o = \mathbf{H}^\top\mathbf{Y}
\quad\Longrightarrow\quad
\mathbf{W}_o = (\mathbf{G} + \lambda\mathbf{I})^{-1}\mathbf{C}.
$$

Since

$$
\mathbf{G} = \sum_{t=1}^{T}\ \sum_{i \in \mathcal{D}_t} \mathbf{h}_i\mathbf{h}_i^\top,
\qquad
\mathbf{C} = \sum_{t=1}^{T}\ \sum_{i \in \mathcal{D}_t} \mathbf{h}_i\mathbf{y}_i^\top,
$$

neither depends on the order in which data arrives or on how it is split into tasks. The head after task $t$ is therefore exactly the head that joint training on tasks $1,\dots,t$ would produce for the same features, so the head itself does not forget.

*Limit:* this holds only because the extractor (backbone, PETL and $\mathbf{W}$) is frozen after Phase 1. Any drift in the features would break the equivalence. Accuracy on old classes can still change as the head redistributes capacity among more classes.

**Property 2: random projection helps only with a nonlinearity.** With a linear projection, $\mathbf{h} = \mathbf{W}^\top\mathbf{f}$ lies in a subspace of dimension $L$ of $\mathbb{R}^M$, so

$$
\operatorname{rank}(\mathbf{G}) = \operatorname{rank}(\mathbf{W}^\top \mathbf{F}^\top \mathbf{F}\, \mathbf{W}) \le L,
$$

where $\mathbf{F} \in \mathbb{R}^{N \times L}$ stacks the backbone features. The ridge head is then equivalent to one fitted on $\mathbf{f}$ directly, and increasing $M$ adds nothing. With the nonlinearity, each coordinate of $\mathbf{h}$ is a nonlinear function of $\mathbf{f}$, so the features are no longer confined to that subspace and the linear head operates on a richer hypothesis class.

*Limit:* the results behind this intuition [8], [9] are probabilistic and asymptotic. They do not guarantee that a particular pre-trained feature distribution becomes more separable. In practice, accuracy rises with $M$ up to a point, at a memory cost of $O(M^2)$ (Section 3.6).

**Property 3: second-order statistics decorrelate the prototypes.** Pre-trained features are highly correlated, so the raw prototypes $\mathbf{c}_k$ (columns of $\mathbf{C}$) of different classes are very similar to each other, and a nearest prototype rule is fragile. Multiplying by $(\mathbf{G} + \lambda\mathbf{I})^{-1}$ suppresses the directions that carry the shared component, which makes the prototypes more distinguishable. The paper relates this to LDA classifiers [VERIFY: figure of distributions of similarity between prototypes before and after decorrelation].

*Limit:* $\mathbf{G}$ is uncentered,

$$
\mathbf{G} = \sum_i \mathbf{h}_i\mathbf{h}_i^\top \;\neq\; \sum_i (\mathbf{h}_i - \boldsymbol{\mu})(\mathbf{h}_i - \boldsymbol{\mu})^\top,
$$

so it is not a covariance matrix, and the link to LDA is an analogy rather than an equivalence. LDA is optimal for Gaussian classes with a shared covariance; RanPAC does not state this assumption and therefore inherits no optimality guarantee from LDA.

### 3.6 Complexity

Phase 2 needs only forward passes. For each sample, RanPAC performs one backbone pass, an $O(LM)$ projection and an $O(M^2)$ update of $\mathbf{G}$. After each task, one $M \times M$ system is solved in $O(M^3)$, repeated for each candidate $\lambda$. Memory is dominated by $\mathbf{G}$ at $O(M^2)$, independent of the number of samples and tasks; $\mathbf{C}$ needs $O(MK)$.

| Method | Training after task 1 | Additional memory | Head cost per task |
|---|---|---|---|
| Prompting [2]–[4] | SGD on prompts every task | Prompt pool | — |
| SLCA [5] | SGD on the backbone every task | $O(KL^2)$, one covariance per class | — |
| NCM / ADaM [6] | None | $O(KL)$ | $O(KL)$ |
| FSA [7] | None | $O(L^2)$ | $O(L^3)$ |
| RanPAC | None | $O(M^2 + MK)$ | $O(M^3)$ per $\lambda$ candidate |

Compared with prompting and SLCA, RanPAC removes all training by gradient descent after the first task. Compared with NCM and FSA, it works in $M$ rather than $L$ dimensions. With $L = 768$ and $M = 10{,}000$, $\mathbf{G}$ stores $10^8$ values, about 400 MB in float32, roughly $170\times$ the size of FSA's covariance matrix. The method is therefore efficient in compute relative to methods trained by SGD, but not in memory relative to the prototype methods it extends.

## 4. Evaluation Methodology

[TODO: complete from the experiment notes]

- **CIL datasets:** CIFAR-100, ImageNet-R, ImageNet-A, CUB-200, OmniBenchmark, VTAB, Stanford Cars [VERIFY list and task splits].
- **DIL datasets:** CORe50, DomainNet [VERIFY].
- **Metric:** final average accuracy over all classes seen [VERIFY whether forgetting is also reported].
- **Baselines:** L2P, DualPrompt, CODA-Prompt, SLCA and ADaM. Joint full fine-tuning on all data serves as the upper bound.
- **Backbones:** ViT-B/16 (main), ResNet and CLIP.

The datasets test different conditions. CIFAR-100 and CUB-200 are close to the ImageNet pre-training distribution. ImageNet-R, ImageNet-A and VTAB involve a strong domain shift, which is where PETL on the first task is expected to matter most.

**Headline results.** RanPAC reduces the relative error by 20–62% compared with ADaM across the seven CIL benchmarks and approaches joint fine-tuning on several datasets. The ablation shows that removing RP, PETL or the second-order statistics each lowers accuracy [TODO: numbers per dataset].

## References

[1] M. D. McDonnell, D. Gong, A. Parvaneh, E. Abbasnejad, and A. van den Hengel, "RanPAC: Random projections and pre-trained models for continual learning," in Proc. NeurIPS, 2023.

[2] Z. Wang et al., "Learning to prompt for continual learning," in Proc. CVPR, 2022.

[3] Z. Wang et al., "DualPrompt: Complementary prompting for rehearsal-free continual learning," in Proc. ECCV, 2022.

[4] J. S. Smith et al., "CODA-Prompt: COntinual decomposed attention-based prompting for rehearsal-free continual learning," in Proc. CVPR, 2023.

[5] G. Zhang, L. Wang, G. Kang, L. Chen, and Y. Wei, "SLCA: Slow learner with classifier alignment for continual learning on a pre-trained model," in Proc. ICCV, 2023.

[6] D.-W. Zhou, H.-J. Ye, D.-C. Zhan, and Z. Liu, "Revisiting class-incremental learning with pre-trained models: Generalizability and adaptivity are all you need," arXiv:2303.07338, 2023.

[7] A. Panos, Y. Kobe, D. O. Reino, R. Aljundi, and R. E. Turner, "First session adaptation: A strong replay-free baseline for class-incremental learning," in Proc. ICCV, 2023.

[8] T. M. Cover, "Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition," IEEE Trans. Electron. Comput., vol. EC-14, no. 3, pp. 326–334, 1965.

[9] A. Rahimi and B. Recht, "Random features for large-scale kernel machines," in Proc. NeurIPS, 2007.

[10] P. Kumari et al., "Continual learning in medical image analysis: A comprehensive review of recent advancements and future prospects," Medical Image Analysis, 2025. [Online]. Available: https://arxiv.org/abs/2312.17004 [VERIFY volume and pages]

[11] "Phân Bón Cà Mau ra mắt AI tính năng chẩn đoán sâu bệnh trên ứng dụng 2Nông," Diễn đàn Doanh nghiệp, Dec. 2024. [Online]. Available: https://diendandoanhnghiep.vn/phan-bon-ca-mau-ra-mat-ai-tinh-nang-chan-doan-sau-benh-tren-ung-dung-2nong-10147460.html [VERIFY date]

[12] "AI 'bác sĩ nông học' trong lòng bàn tay," VietNamNet. [Online]. Available: https://vietnamnet.vn/ai-bac-si-nong-hoc-trong-long-ban-tay-2358587.html [VERIFY title and date]
