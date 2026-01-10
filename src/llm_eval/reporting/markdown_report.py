"""
Markdown report generation for llm-eval.

This report is designed for:
- Human review
- GitHub rendering
- Portfolio demonstration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape


class MarkdownReport:
    """
    Generates a human-readable Markdown evaluation report.
    """

    def __init__(self, output_dir: Path, template_dir: Path | None = None) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        template_dir = template_dir or Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(self, results: Dict[str, Any]) -> Path:
        """
        Render the Markdown report using Jinja2 templates.
        """
        template = self.env.get_template("report.md.j2")
        content = template.render(results=results)

        report_path = self.output_dir / "evaluation_report.md"
        report_path.write_text(content, encoding="utf-8")

        return report_path
