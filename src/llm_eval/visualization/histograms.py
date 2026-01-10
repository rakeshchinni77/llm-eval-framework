"""
Histogram visualizations for metric score distributions.

Purpose:
- Show how scores are distributed per metric
- Identify skew, variance, and outliers
"""

from pathlib import Path
from typing import Dict, List
import json

import matplotlib.pyplot as plt


def plot_histograms(
    *,
    scores: Dict[str, List[float]],
    output_dir: Path,
    bins: int = 20,
) -> None:
    """
    Generate histogram plots for each metric.

    Args:
        scores: {metric_name: [scores]}
        output_dir: directory to save PNGs
        bins: number of histogram bins
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, values in scores.items():
        if not values:
            # Skip empty metrics safely
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=bins)
        plt.title(f"{metric.replace('_', ' ').title()} Score Distribution")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.xlim(0.0, 1.0)
        plt.grid(True, linestyle="--", alpha=0.5)

        path = output_dir / f"histogram_{metric}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()


def generate_histograms(
    results_dir: Path,
    screenshots_dir: Path,
) -> None:
    """
    Load raw scores from evaluation output and generate histograms.

    Expected file:
    results/raw_scores.json
    """
    raw_scores_path = results_dir / "raw_scores.json"

    if not raw_scores_path.exists():
        raise FileNotFoundError(
            "raw_scores.json not found. Run evaluation first."
        )

    with raw_scores_path.open("r", encoding="utf-8") as f:
        raw_scores = json.load(f)

    screenshots_dir.mkdir(parents=True, exist_ok=True)

    # Generate histograms per model
    for model_name, metrics in raw_scores.items():
        plot_histograms(
            scores=metrics,
            output_dir=screenshots_dir,
        )


# SCRIPT ENTRYPOINT — THIS IS WHAT WAS MISSING BEFORE
if __name__ == "__main__":
    results_dir = Path("results")
    screenshots_dir = Path("screenshots")

    with open(results_dir / "raw_scores.json", "r", encoding="utf-8") as f:
        raw_scores = json.load(f)

    screenshots_dir.mkdir(exist_ok=True)

    # Single-model visualization (correct for now)
    model_name = list(raw_scores.keys())[0]

    plot_histograms(
        scores=raw_scores[model_name],
        output_dir=screenshots_dir,
    )
