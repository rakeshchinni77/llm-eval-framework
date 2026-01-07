"""
CLI entry point for llm-eval.

This module defines the Typer-based command-line interface used for
running LLM evaluation workflows in local, CI/CD, and Docker environments.
"""

from pathlib import Path

import typer
from rich.console import Console

from llm_eval.version import __version__
from llm_eval.config.loader import load_config, ConfigLoadError

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
        help="Directory where evaluation reports and visualizations will be saved.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging output.",
    ),
) -> None:
    """
    Run an evaluation based on the provided configuration file.

    Responsibilities:
    - Validate configuration
    - Prepare evaluation context
    - (Evaluation runner will be plugged in later)
    """
    try:
        # Load & validate config (EXIT CODE 1 on failure)
        cfg = load_config(config)

        if verbose:
            console.print("[bold cyan]Verbose mode enabled[/bold cyan]")

        console.print(
            f"[bold green]llm-eval v{__version__}[/bold green] configuration loaded"
        )
        console.print(f"Models: {[m.name for m in cfg.models]}")
        console.print(f"Output directory: [yellow]{output_dir}[/yellow]")

        # Placeholder – evaluation runner will be added in next phases
        console.print(
            "[bold blue]Configuration validated successfully.[/bold blue]"
        )

        # SUCCESS
        raise typer.Exit(code=0)

    except ConfigLoadError as exc:
        console.print(
            f"[bold red]Configuration error:[/bold red]\n{exc}",
            highlight=False,
        )
        # EXIT CODE 1 = Config error
        raise typer.Exit(code=1)

    except Exception as exc:  # pragma: no cover (future runtime errors)
        console.print(
            f"[bold red]Fatal error:[/bold red] {exc}",
            highlight=False,
        )
        # EXIT CODE 2 = Metric / runtime failure
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
