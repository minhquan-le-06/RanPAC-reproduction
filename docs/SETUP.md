# RanPAC setup on Colab

Runs the Phase 2 ablations on Split CIFAR-100 with seed `1993` (10 tasks, 10 classes each).

All three ablation notebooks have already been run; the results are in the W&B project `pnp123/ML`.

## Installation

1. Copy the repository (at least `src/` and `tests/`) to `MyDrive/RanPAC_implement`. If you place it elsewhere, edit `SOURCE_DIR` in the notebook. The notebooks add `SOURCE_DIR/src` to `sys.path`.
2. Open the notebook in Google Colab and choose **Runtime → Change runtime type → GPU**. The two setup cells install `timm==1.0.22` and `wandb==0.31.0` and mount Drive. PyTorch and torchvision come from the Colab runtime. CIFAR-100 and the ViT weights are downloaded automatically when embeddings are extracted.
3. Add `WANDB_API_KEY` to **Colab Secrets** and grant the notebook access. All three notebooks log online to `pnp123/ML` (`WANDB_ENTITY = 'pnp123'`, `WANDB_PROJECT = 'ML'`); use an API key with write access to that project, or change `WANDB_ENTITY` and `WANDB_PROJECT` to your own.

## Notebook configuration

| Notebook | `PRESET` | `RESUME` |
| --- | --- | --- |
| [01_ncm_cosine.ipynb](../notebooks/01_ncm_cosine.ipynb) | `ncm`: NCM cosine | `False` |
| [02_ridge_no_rp.ipynb](../notebooks/02_ridge_no_rp.ipynb) | `ridge`: ridge without random projection | `False` |
| [03_ridge_rp_relu.ipynb](../notebooks/03_ridge_rp_relu.ipynb) | `rp-table1`: RP + ReLU, `M=10000`, per-task `λ` selection | `True` |

All three use `STOP_AFTER_TASK = 10`, `RUN_LABEL = 'main'`, `BATCH_SIZE = 48`, `NUM_WORKERS = 2` and `STATS_DEVICE = 'cpu'`. The GPU is used for embedding extraction. `RUN_WANDB_SMOKE = False` by default.

## Fresh run or resume

- **Fresh run**: set `RESUME = False` and change `RUN_LABEL` from `main` to an unused name for that preset (e.g. `rerun-01`) to avoid clashing with existing checkpoint folders and W&B run names.
- **Resume**: keep `RESUME = True` and reuse the same `WORK_DIR`, `RUN_LABEL`, preset, source code, configuration and W&B project as the original run. For notebook `03` as configured, the checkpoint must exist at `MyDrive/ranpac-runs/runs/rp-table1-s1993-main/checkpoint.pt`.
- Run the cells in order. To test W&B, set `RUN_WANDB_SMOKE = True` and confirm `smoke/ok = 1` at the printed URL. To try a single task on a fresh run, set `STOP_AFTER_TASK = 1`, then set `RESUME = True` and raise `STOP_AFTER_TASK` to continue.

## Outputs

By default runs are stored on Drive at `MyDrive/ranpac-runs/runs/<preset>-s1993-<run_label>/`, containing `checkpoint.pt` (for resuming), `config.json`, `metrics.csv` (`A[t]`) and `accuracy_matrix.csv` (`R[t,i]`). The shared embedding cache is in `MyDrive/ranpac-runs/embeddings/`.

## Full Algorithm 1

Open [04_ranpac_full.ipynb](../notebooks/04_ranpac_full.ipynb); `src/ranpac_full.py` must be in `SOURCE_DIR/src` along with the modules above. The notebook trains an AdaptFormer on task 1, freezes the adapted backbone and runs Phase 2 from task 1. By default it trains 20 epochs and runs all 10 tasks with `M=10000`. Set `STOP_AFTER_TASK = 2` for a Phase 2 pilot, then raise it to 10 and enable `RESUME` to continue.

The full run is stored in `MyDrive/ranpac-runs/runs/full-s1993-<run_label>/` and adds `phase1_metrics.csv`. The notebook documents how to continue from an epoch checkpoint, the VRAM-reduction options, and the differences from the authors' configuration.

## `M` sweep on CIFAR-100

Open [05_ranpac_m_sweep.ipynb](../notebooks/05_ranpac_m_sweep.ipynb); `src/ranpac_core.py`, `src/ranpac_data.py` and `src/ranpac_colab.py` must be in `SOURCE_DIR/src`. If you already built the cache with notebook 03, reuse the same `WORK_DIR` and library versions to reuse the embeddings. The notebook runs `M = 300, 500, 1000, 1250, 2500, 10000` in turn and produces a table comparing accuracy, time and RAM.

## Algorithm 1 on ImageNet-R

Open [06_ranpac_full_imagenetr.ipynb](../notebooks/06_ranpac_full_imagenetr.ipynb) in Colab with a GPU runtime. Place the files as in the table below, or edit the paths in the Drive-mount cell. The source folder is shared with notebook 04.

| Default path | Notebook variable |
| --- | --- |
| `MyDrive/RanPAC_implement/RanPAC-reimplement-imagenetr.zip` (a zip of this repository; the notebook adds its `src/` folder to `sys.path`) | `SOURCE_ZIP` |
| `MyDrive/imagenet-r.zip`, with the same `train/` and `test/` split as the authors' notebook. The notebook downloads it if missing. | `DATASET_INPUT` |
| `MyDrive/ranpac-runs/` for checkpoints and results | `WORK_DIR` |

W&B logs online to `pnp123/ML` by default, as in notebook 04; grant the notebook read access to the Colab secret `WANDB_API_KEY`. To try one epoch, set `STOP_AFTER_EPOCH = 1` and keep `PHASE1_EPOCHS = 20`. Then set `RESUME = True`, `STOP_AFTER_EPOCH = None` and keep the same run label, W&B mode and training configuration.

The last cell reads the authors' preset 7 results from `MyDrive/RanPAC-original-imagenetr-results/` to build the comparison table and plot. Run it after both runs have finished all 10 tasks.
