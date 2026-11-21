"""Read final CIFAR-100 M-sweep checkpoints and export numerical ranks of G."""

import argparse
import csv
from pathlib import Path

import torch

from ranpac_colab import SWEEP_DIMS
from ranpac_core import NUM_TASKS, SEED
from ranpac_data import atomic_output


def read_rank(path, dim):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    tasks = [record["task"] for record in checkpoint["records"]]
    if tasks != list(range(1, NUM_TASKS + 1)):
        raise ValueError(f"{path}: expected completed tasks 1 through {NUM_TASKS}, got {tasks}")

    gram = checkpoint["model"]["G"]
    if gram.shape != (dim, dim):
        raise ValueError(f"{path}: expected G with shape ({dim}, {dim}), got {tuple(gram.shape)}")

    # G stores H.T @ H before the ridge penalty. It is symmetric.
    rtol = dim * torch.finfo(gram.dtype).eps
    rank = torch.linalg.matrix_rank(gram, hermitian=True, atol=0.0, rtol=rtol).item()
    return {
        "M": dim,
        "task": tasks[-1],
        "rank_g_relu": rank,
        "g_dtype": str(gram.dtype),
        "atol": 0.0,
        "rtol": rtol,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path, help="Directory containing the runs/ folder")
    parser.add_argument("--run-label", default="m-sweep-01", help="RUN_LABEL used in notebook 05")
    args = parser.parse_args()
    label = args.run_label
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in label):
        parser.error("--run-label must use lowercase letters, digits, hyphens or underscores")

    checkpoints = {
        dim: args.work_dir / "runs" / f"rp-sweep-m{dim}-s{SEED}-{label}" / "checkpoint.pt"
        for dim in SWEEP_DIMS
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        parser.exit(1, "Missing checkpoints:\n" + "\n".join(missing) + "\n")

    rows = []
    for dim, path in checkpoints.items():
        print(f"Computing rank(G) for M={dim}", flush=True)
        row = read_rank(path, dim)
        rows.append(row)
        print(f"M={dim}: rank(G)={row['rank_g_relu']}", flush=True)

    output = args.work_dir / "sweeps" / f"m-sweep-s{SEED}-{label}" / "sweep_rank_g.csv"
    with atomic_output(output) as temp, temp.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
