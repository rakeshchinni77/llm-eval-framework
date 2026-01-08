"""
Benchmark dataset loader.

Supports JSONL and CSV formats with strict validation.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

from llm_eval.data.validators import validate_row


class DatasetLoadError(RuntimeError):
    """Raised when dataset loading or validation fails."""


def load_dataset(path: Path) -> List[Dict]:
    """
    Load and validate benchmark dataset.

    :param path: Path to JSONL or CSV dataset
    :return: List of validated dataset rows
    """
    try:
        if path.suffix.lower() == ".jsonl":
            df = pd.read_json(path, lines=True)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raise DatasetLoadError("Dataset must be JSONL or CSV")

    except Exception as exc:
        raise DatasetLoadError(f"Failed to read dataset: {exc}") from exc

    records = df.to_dict(orient="records")

    if not records:
        raise DatasetLoadError("Dataset is empty")

    for idx, row in enumerate(records, start=1):
        try:
            validate_row(row)
        except ValueError as exc:
            raise DatasetLoadError(
                f"Row {idx} failed validation: {exc}"
            ) from exc

    return records
