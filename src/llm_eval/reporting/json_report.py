"""
JSON report generation for llm-eval.

This report is designed for:
- CI/CD parsing
- Programmatic analysis
- Artifact storage
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class JSONReport:
    """
    Generates a machine-readable JSON evaluation report.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, results: Dict[str, Any]) -> Path:
        """
        Write evaluation results to a JSON file.

        Expected structure of `results`:
        {
            "metadata": {...},
            "models": {...},
            "metrics": {...},
            "aggregates": {...},
            "quality_gates": {...}
        }
        """
        report_path = self.output_dir / "evaluation_report.json"

        with report_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return report_path
