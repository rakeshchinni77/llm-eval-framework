"""
Dataset row validation utilities.

These validators ensure benchmark datasets meet strict schema requirements
expected by the evaluation pipeline.
"""

from typing import Dict, List


REQUIRED_FIELDS = {
    "id",
    "query",
    "expected_answer",
    "retrieved_contexts",
    "difficulty",
    "category",
}


ALLOWED_DIFFICULTY = {"easy", "medium", "hard"}


def validate_row(row: Dict) -> None:
    """
    Validate a single dataset row.

    Raises ValueError on any schema violation.
    """
    missing = REQUIRED_FIELDS - row.keys()
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    if not isinstance(row["id"], str) or not row["id"]:
        raise ValueError("Field 'id' must be a non-empty string")

    if not isinstance(row["query"], str) or not row["query"]:
        raise ValueError("Field 'query' must be a non-empty string")

    if not isinstance(row["expected_answer"], str):
        raise ValueError("Field 'expected_answer' must be a string")

    if not isinstance(row["retrieved_contexts"], list):
        raise ValueError("Field 'retrieved_contexts' must be a list")

    for ctx in row["retrieved_contexts"]:
        if not isinstance(ctx, str):
            raise ValueError("Each retrieved_context must be a string")

    if row["difficulty"] not in ALLOWED_DIFFICULTY:
        raise ValueError(
            f"Invalid difficulty '{row['difficulty']}'. "
            f"Must be one of {sorted(ALLOWED_DIFFICULTY)}"
        )

    if not isinstance(row["category"], str) or not row["category"]:
        raise ValueError("Field 'category' must be a non-empty string")
