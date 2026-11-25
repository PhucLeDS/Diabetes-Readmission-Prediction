"""Normalize the approved raw markers and audit all encounter columns."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.io_utils import CsvValue, read_json, write_csv
from src.paths import CONFIG_DIR, INTERIM_DIR, PREPARATION_OUTPUT_DIR
from src.preparation.data_loading import COUNT_COLUMNS, ENCOUNTER_COLUMNS, ID_COLUMNS, TARGET_COLUMN, RawTable
from src.validation import validate_schema


CLEANING_CONFIG_PATH = CONFIG_DIR / "preparation.json"
CLEANED_DATA_PATH = INTERIM_DIR / "cleaned.csv"
CLEANING_AUDIT_PATH = PREPARATION_OUTPUT_DIR / "cleaning_audit.csv"
AUDIT_COLUMNS = (
    "column", "rows_before", "rows_after", "source_marker", "marker_before",
    "marker_after", "empty_before", "empty_after", "missing_rate_before",
    "missing_rate_after", "changed_cells", "preserved_literal", "literal_before",
    "literal_after",
)


@dataclass
class CleaningResult:
    """Cleaned records in source order and an audit row for each source column."""

    columns: tuple[str, ...]
    rows: list[dict[str, str]]
    audit: list[dict[str, CsvValue]]


def _read_cleaning_rules(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    policy = read_json(path)["cleaning"]
    markers = policy["missing_markers"]
    literals = policy["preserve_literals"]
    eligible = set(ENCOUNTER_COLUMNS) - {*ID_COLUMNS, *COUNT_COLUMNS, TARGET_COLUMN}
    if not isinstance(markers, dict) or not isinstance(literals, dict) or not markers:
        raise ValueError(f"{path}: expected nonempty missing-marker rules and a literal mapping.")
    if not eligible.issuperset(markers.keys() | literals.keys()) or markers.keys() & literals.keys():
        raise ValueError(f"{path}: rules must use distinct, eligible categorical columns.")
    for column, marker in markers.items():
        if marker != ("Unknown/Invalid" if column == "gender" else "?"):
            raise ValueError(f"{path}: invalid cleaning marker for {column!r}.")
    if any(not isinstance(literal, str) or not literal for literal in literals.values()):
        raise ValueError(f"{path}: preserved literals must be nonempty strings.")
    return markers, literals


def clean_encounters(
    table: RawTable, *, config_path: Path = CLEANING_CONFIG_PATH,
) -> CleaningResult:
    """Apply the configured replacements to a verified encounter table."""
    validate_schema(table.columns, expected=ENCOUNTER_COLUMNS)
    markers, literals = _read_cleaning_rules(config_path)
    empty_before: Counter[str] = Counter()
    changed: Counter[str] = Counter()
    preserved: Counter[str] = Counter()
    rows = []
    for row_number, source in enumerate(table.rows, start=1):
        row = source.copy()
        for column, value in source.items():
            if value == "":
                empty_before[column] += 1
            elif markers.get(column) == value:
                row[column] = ""
                changed[column] += 1
            elif value == "?":
                raise ValueError(f"Unapproved '?' marker in column {column!r}, data row {row_number}.")
            if literals.get(column) == value:
                preserved[column] += 1
        rows.append(row)

    count = len(rows)
    audit = [
        {
            "column": column,
            "rows_before": len(table.rows),
            "rows_after": count,
            "source_marker": markers.get(column, ""),
            "marker_before": changed[column],
            "marker_after": 0,
            "empty_before": empty_before[column],
            "empty_after": empty_before[column] + changed[column],
            "missing_rate_before": (empty_before[column] + changed[column]) / count if count else None,
            "missing_rate_after": (empty_before[column] + changed[column]) / count if count else None,
            "changed_cells": changed[column],
            "preserved_literal": literals.get(column, ""),
            "literal_before": preserved[column],
            "literal_after": preserved[column],
        }
        for column in table.columns
    ]
    return CleaningResult(columns=tuple(table.columns), rows=rows, audit=audit)


def save_cleaning_result(
    result: CleaningResult,
    *,
    cleaned_path: Path = CLEANED_DATA_PATH,
    audit_path: Path = CLEANING_AUDIT_PATH,
) -> tuple[Path, Path]:
    """Write the cleaned dataset and audit with the project's CSV conventions."""
    write_csv(cleaned_path, result.columns, result.rows)
    write_csv(audit_path, AUDIT_COLUMNS, result.audit)
    return cleaned_path, audit_path
