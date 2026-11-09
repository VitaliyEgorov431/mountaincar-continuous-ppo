"""Produce training and test-comparison figures without a display."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(directory):
    directory = Path(directory)
    runs = sorted(directory.glob("final/seed_*"))
    if not runs and (directory / "validation.csv").exists():
        runs = [directory]
    if runs:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for run in runs:
            with (run / "validation.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            for ax, metric, label in zip(axes, ("mean_reward", "success_rate"), ("Validation mean reward", "Validation success rate")):
                ax.plot([int(r["step"]) for r in rows], [float(r[metric]) for r in rows], label=run.name)
                ax.set(xlabel="Environment transitions", ylabel=label)
                if metric == "success_rate":
                    ax.set_ylim(0, 1.05)
                ax.grid(alpha=0.3)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(directory / "training_curves.png", dpi=150)
        plt.close(fig)
    results_file = directory / "test_results.json"
    if results_file.exists():
        results = json.loads(results_file.read_text(encoding="utf-8"))
        models = [v["summary"] for k, v in results.items() if k.startswith("seed_")]
        groups = {"PPO": models, "Random": [results["random"]["summary"]]}
        if "random_hold" in results:
            groups["Random hold"] = [results["random_hold"]["summary"]]
        groups["Heuristic"] = [results["heuristic"]["summary"]]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric, ylabel in zip(axes, ("mean_reward", "success_rate"), ("Mean reward", "Success rate")):
            values = [[x[metric] for x in group] for group in groups.values()]
            means = [np.mean(x) for x in values]
            errors = [np.std(x, ddof=1) if len(x) > 1 else 0 for x in values]
            ax.bar(list(groups), means, yerr=errors, capsize=4)
            ax.set_ylabel(ylabel)
            if metric == "success_rate":
                ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("PPO error bars: sample SD across training seeds")
        fig.tight_layout()
        fig.savefig(directory / "comparison.png", dpi=150)
        plt.close(fig)
