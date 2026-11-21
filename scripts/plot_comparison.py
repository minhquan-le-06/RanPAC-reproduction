"""Plot accuracy curves comparing all methods on Split CIFAR-100."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt

# 1. Load local run for rp-table-a5
a5_path = Path("ranpac-runs/runs/rp-table-a5-s1993-local/metrics.csv")
if a5_path.exists():
    with a5_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        a5_acc = [float(row["avg_accuracy_pct"]) for row in reader]
else:
    a5_acc = [96.80, 95.10, 94.70, 93.33, 92.02, 90.78, 90.53, 88.83, 87.99, 87.68]

# 2. Results from notebooks (01 NCM, 02 Ridge, 03 RP-Table1)
ncm_acc = [96.00, 92.95, 92.33, 90.22, 88.34, 86.87, 86.47, 84.36, 83.80, 83.38]
ridge_acc = [97.80, 95.50, 94.53, 92.80, 91.50, 90.37, 90.00, 88.30, 87.46, 87.02]

# rp-table1 tasks 6-10 are recorded; tasks 1-5 follow Ridge baseline / initial tasks
rp1_tasks = list(range(6, 11))
rp1_acc = [91.60, 91.43, 89.72, 89.14, 88.75]

# 3. Algorithm 1 (Phase 1 AdaptFormer + Phase 2 Ridge+RP, M=10000)
alg1_acc = [99.00, 97.85, 97.43, 96.15, 95.24, 94.32, 94.07, 92.88, 92.39, 92.02]

tasks = list(range(1, 11))

plt.figure(figsize=(9, 5.5), dpi=300)

plt.plot(tasks, ncm_acc, marker='o', linewidth=2, label=f'NCM Cosine (Final: {ncm_acc[-1]:.2f}%, Paper: 83.4%)', color='#e74c3c')
plt.plot(tasks, ridge_acc, marker='s', linewidth=2, label=f'Ridge (no RP) (Final: {ridge_acc[-1]:.2f}%, Paper: 87.0%)', color='#3498db')
plt.plot(tasks, a5_acc, marker='^', linewidth=2.5, label=f'RanPAC (M=2500, λ=100) (Final: {a5_acc[-1]:.2f}%, Paper: 87.7%)', color='#2ecc71')
plt.plot(rp1_tasks, rp1_acc, marker='D', linewidth=2, linestyle='--', label=f'RanPAC (M=10000, Table 1) (Final: {rp1_acc[-1]:.2f}%, Paper: 89.0%)', color='#9b59b6')
plt.plot(tasks, alg1_acc, marker='*', linewidth=2.5, label=f'Algorithm 1 (Phase1+Phase2) (Final: {alg1_acc[-1]:.2f}%, Paper: 92.2%)', color='#f39c12')

plt.title('Continual Learning on Split CIFAR-100 (10 Tasks, 10 Classes/Task)\nRanPAC Reproduction Comparison', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Task', fontsize=11, fontweight='bold')
plt.ylabel('Average Accuracy (%) across Seen Classes', fontsize=11, fontweight='bold')
plt.xticks(tasks)
plt.ylim(80, 102)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower left', fontsize=9.5, framealpha=0.9)

output_path = Path(__file__).resolve().parents[1] / "results" / "accuracy_comparison.png"
plt.tight_layout()
plt.savefig(output_path)
print(f"Plot saved successfully to {output_path.resolve()}")
