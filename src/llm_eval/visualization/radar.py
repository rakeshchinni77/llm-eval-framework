"""
Radar chart visualization for comparing models across metrics.

Purpose:
- Quick visual comparison of model strengths
- Consistent scale across metrics (0–1)
"""

from math import pi
from pathlib import Path
from typing import Dict
import json

import matplotlib.pyplot as plt


def plot_radar(
    *,
    model_scores: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
) -> None:
    """
    Generate radar chart comparing models.

    Args:
        model_scores: {model: {metric: {mean, median, std, ...}}}
        output_dir: directory to save PNG
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract metric names
    metrics = list(next(iter(model_scores.values())).keys())
    num_metrics = len(metrics)

    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], metrics)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"])
    plt.ylim(0.0, 1.0)

    for model_name, metric_stats in model_scores.items():
        # USE MEAN ONLY
        values = [stats["mean"] for stats in metric_stats.values()]
        values += values[:1]

        ax.plot(angles, values, linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.1)

    plt.title("Model Comparison Across Metrics", pad=20)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    path = output_dir / "radar_model_comparison.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ENTRYPOINT — REQUIRED FOR SCREENSHOT GENERATION
if __name__ == "__main__":
    results_dir = Path("results")
    screenshots_dir = Path("screenshots")

    aggregates_path = results_dir / "aggregates.json"

    if not aggregates_path.exists():
        raise FileNotFoundError("aggregates.json not found. Run evaluation first.")

    with aggregates_path.open("r", encoding="utf-8") as f:
        aggregates = json.load(f)

    screenshots_dir.mkdir(exist_ok=True)

    plot_radar(
        model_scores=aggregates,
        output_dir=screenshots_dir,
    )
