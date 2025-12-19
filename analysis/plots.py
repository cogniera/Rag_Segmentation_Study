import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


# Output directory
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def plot_recall_comparison():
    baseline_path = Path("results/baseline/metrics.json")
    fitted_path = Path("results/query_fitted/metrics.json")

    if not baseline_path.exists() or not fitted_path.exists():
        return

    with open(baseline_path) as f:
        baseline_metrics = json.load(f)

    with open(fitted_path) as f:
        fitted_metrics = json.load(f)

    # Convert keys to ints for plotting
    ks = sorted(int(k) for k in baseline_metrics.keys())

    baseline_recall = [
        baseline_metrics[str(k)]["reranked_recall"] for k in ks
    ]

    fitted_recall = [
        fitted_metrics[str(k)]["reranked_recall"] for k in ks
    ]

    plt.figure()
    plt.plot(ks, baseline_recall, label="Baseline", marker="o")
    plt.plot(ks, fitted_recall, label="Query-fitted", marker="o")
    plt.xlabel("k")
    plt.ylabel("End-to-end Recall@k")
    plt.title("Recall comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig(FIG_DIR / "recall_comparison.png")
    plt.close()


if __name__ == "__main__":
    plot_recall_comparison()