"""Run RanPAC Phase 2 locally (Windows / Linux, CUDA required).

Usage:
    python run_local.py --preset ncm --wandb_entity YOUR_ENTITY --wandb_project ML
    python run_local.py --preset ridge --wandb_entity YOUR_ENTITY --wandb_project ML
    python run_local.py --preset rp-table1 --wandb_entity YOUR_ENTITY --wandb_project ML
    python run_local.py --preset rp-table-a5 --wandb_entity YOUR_ENTITY --wandb_project ML

Presets and paper benchmarks (Table 1 / Table A5):
    ncm          NCM cosine            A[10] = 83.4%
    ridge        Ridge no RP           A[10] = 87.0%
    rp-table1    Ridge + RP M=10000    A[10] = 89.0%
    rp-table-a5  Ridge + RP M=2500     A[10] = 87.7%  (Table A5)

Results (CSV + checkpoint) are saved to ./ranpac-runs/ by default.
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Patch ranpac_colab so it works without Colab / with a local WANDB_API_KEY.
# ---------------------------------------------------------------------------

def _local_configure_wandb(entity, mode):
    """Drop-in replacement for ranpac_colab.configure_wandb (no Colab dep)."""
    if not entity or mode not in {"online", "offline"}:
        raise ValueError("Set a valid W&B entity and mode (online/offline).")
    os.environ["WANDB_DISABLE_CODE"] = "true"
    # On Colab the key comes from userdata; locally it must be in the env.
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "Set the WANDB_API_KEY environment variable before running.\n"
            "  Windows CMD:        set WANDB_API_KEY=your_key\n"
            "  Windows PowerShell: $env:WANDB_API_KEY='your_key'\n"
            "  Or add it to a .env file and use python-dotenv."
        )
    import wandb
    return wandb


def _local_run_phase2_no_gpu_check(preset, work_dir, **kwargs):
    """Call run_phase2 after removing the hard GPU requirement."""
    import ranpac_colab as _rc
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. RanPAC requires a CUDA-capable GPU.\n"
            "If you have an NVIDIA GPU, install the matching CUDA PyTorch build:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        )

    # Monkey-patch the Colab-specific W&B setup with our local version.
    _rc.configure_wandb = _local_configure_wandb
    return _rc.run_phase2(preset, work_dir, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Run RanPAC Phase 2 locally.")
    parser.add_argument("--preset", required=True,
                        choices=["ncm", "ridge", "rp-table1", "rp-table-a5"],
                        help="Which experiment to run.")
    parser.add_argument("--wandb_entity", required=True,
                        help="Your W&B username or team slug.")
    parser.add_argument("--wandb_project", default="ML",
                        help="W&B project name (default: ML).")
    parser.add_argument("--wandb_mode", default="online",
                        choices=["online", "offline"],
                        help="W&B logging mode (default: online).")
    parser.add_argument("--run_label", default="local",
                        help="Short label appended to the run name (default: local).")
    parser.add_argument("--work_dir", default="ranpac-runs",
                        help="Directory for checkpoints and embeddings (default: ranpac-runs).")
    parser.add_argument("--data_root", default="ranpac-data",
                        help="Directory where CIFAR-100 will be downloaded (default: ranpac-data).")
    parser.add_argument("--stats_device", default="cpu", choices=["cpu", "cuda"],
                        help="Device for accumulating statistics (default: cpu).")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for ViT feature extraction (default: 64).")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default: 4). Use 0 on Windows if you get errors.")
    parser.add_argument("--stop_after_task", type=int, default=10,
                        help="Stop after this task number 1-10 (default: 10).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted run (keep same preset + run_label).")
    args = parser.parse_args()

    # Make sure the source files are importable.
    script_dir = Path(__file__).parent.resolve()
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    work_dir = Path(args.work_dir).resolve()

    print(f"Preset  : {args.preset}")
    print(f"W&B     : {args.wandb_entity}/{args.wandb_project} ({args.wandb_mode})")
    print(f"Work dir: {work_dir}")
    print(f"Data dir: {Path(args.data_root).resolve()}")
    print()

    records = _local_run_phase2_no_gpu_check(
        args.preset,
        work_dir,
        entity=args.wandb_entity,
        project=args.wandb_project,
        stop_after_task=args.stop_after_task,
        run_label=args.run_label,
        resume=args.resume,
        wandb_mode=args.wandb_mode,
        batch_size=args.batch_size,
        workers=args.workers,
        stats_device=args.stats_device,
        data_root=args.data_root,
    )
    return records


if __name__ == "__main__":
    main()
