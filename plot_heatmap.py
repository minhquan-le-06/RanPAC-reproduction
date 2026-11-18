"""Plot forgetting heatmap matrix R[t, i] from accuracy_matrix.csv."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

matrix_file = Path("ranpac-runs/runs/rp-table-a5-s1993-local/accuracy_matrix.csv")

if not matrix_file.exists():
    raise FileNotFoundError(f"Missing {matrix_file}")

# Load 10x10 accuracy matrix R[train_task, test_task]
R = np.full((10, 10), np.nan)

with matrix_file.open(encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = int(row["train_task"]) - 1
        i = int(row["test_task"]) - 1
        R[t, i] = float(row["accuracy_pct"])

plt.figure(figsize=(8.5, 7), dpi=300)

# Mask upper triangle (tasks not seen yet)
masked_R = np.ma.masked_invalid(R)
cmap = plt.cm.Blues.copy()
cmap.set_bad(color="whitesmoke")

im = plt.imshow(masked_R, cmap=cmap, vmin=80, vmax=100)
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label("Accuracy (%)", fontsize=11, fontweight="bold")

# Annotate each cell with accuracy value
for t in range(10):
    for i in range(10):
        if not np.isnan(R[t, i]):
            color = "white" if R[t, i] >= 91 else "black"
            plt.text(i, t, f"{R[t, i]:.1f}%", ha="center", va="center", color=color, fontsize=9.5)

plt.title("RanPAC (Table A5: M=2500, λ=100)\nTask Accuracy Matrix R[t, i] & Catastrophic Forgetting Evaluation",
          fontsize=12, fontweight="bold", pad=12)
plt.xlabel("Evaluated Task (i)", fontsize=11, fontweight="bold")
plt.ylabel("Training Task (t)", fontsize=11, fontweight="bold")
plt.xticks(range(10), [f"Task {k}" for k in range(1, 11)])
plt.yticks(range(10), [f"After T{k}" for k in range(1, 11)])

output_path = Path("forgetting_heatmap.png")
plt.tight_layout()
plt.savefig(output_path)
print(f"Heatmap saved successfully to {output_path.resolve()}")
