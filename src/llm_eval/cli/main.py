"""
CLI entry point for llm-eval.

This module defines the Typer-based command-line interface used for
running LLM evaluation workflows in local, CI/CD, and Docker environments.
"""

from pathlib import Path

import typer
from rich.console import Console

import llm_eval.metrics

from llm_eval.version import __version__
from llm_eval.config.loader import load_config, ConfigLoadError
from llm_eval.data.dataset_loader import load_dataset
from llm_eval.evaluation.runner import EvaluationRunner

app = typer.Typer(
    name="llm-eval",
    help="Production-grade LLM evaluation framework with multi-metric analysis.",
    no_args_is_help=True,
)

console = Console()


@app.command()
def run(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        readable=True,
        help="Path to evaluation configuration file (YAML or JSON).",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Directory where evaluation results and visualizations will be saved.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging output.",
    ),
) -> None:
    """
    Run the full LLM evaluation pipeline.

    Responsibilities:
    - Load and validate configuration
    - Load dataset
    - Execute evaluation runner
    - Persist raw scores for reporting & visualization
    """
    try:
        cfg = load_config(config)

        if verbose:
            console.print("[bold cyan]Verbose mode enabled[/bold cyan]")

        console.print(f"[bold green]llm-eval v{__version__}[/bold green]")
        console.print(f"Dataset: {cfg.dataset.path}")
        console.print(f"Models: {[m.name for m in cfg.models]}")
        console.print(f"Output directory: [yellow]{output_dir}[/yellow]")

        dataset = load_dataset(cfg.dataset.path)

        runner = EvaluationRunner(
            dataset=dataset,
            models=[m.model_dump() for m in cfg.models],
            metrics=cfg.metrics,
            output_dir=output_dir,
        )

        runner.run()

        console.print("[bold blue]Evaluation completed successfully[/bold blue]")
        raise typer.Exit(code=0)

    except ConfigLoadError as exc:
        console.print(
            f"[bold red]Configuration error:[/bold red]\n{exc}",
            highlight=False,
        )
        raise typer.Exit(code=1)

    except typer.Exit:
        # Allow Typer exits to propagate cleanly
        raise

    except Exception as exc:  # pragma: no cover
        console.print(
            f"[bold red]Fatal error:[/bold red] {exc}",
            highlight=False,
        )
        raise typer.Exit(code=2)


@app.command()
def version() -> None:
    """
    Print the installed llm-eval version.
    """
    console.print(f"llm-eval version: [bold]{__version__}[/bold]")


def main() -> None:
    """
    Entrypoint for console_scripts.
    """
    app()


if __name__ == "__main__":
    main()
