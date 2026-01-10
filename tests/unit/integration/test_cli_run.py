import subprocess
from pathlib import Path


def test_full_cli_run(tmp_path):
    config = Path("examples/config_ci.yaml")
    output_dir = tmp_path / "results"

    result = subprocess.run(
        [
            "poetry",
            "run",
            "llm-eval",
            "run",
            "--config",
            str(config),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (output_dir / "raw_scores.json").exists()
    assert (output_dir / "aggregates.json").exists()
