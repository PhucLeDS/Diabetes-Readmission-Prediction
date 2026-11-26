"""Normalize the approved raw markers and audit all encounter columns."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.io_utils import read_json, write_csv
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

    frame: pd.DataFrame
    audit: pd.DataFrame


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
    source = table.frame
    validate_schema(list(source.columns), expected=ENCOUNTER_COLUMNS)
    markers, literals = _read_cleaning_rules(config_path)
    cleaned = source.copy()
    count = len(source)
    audit = []
    for column, values in source.items():
        marker = markers.get(column, "")
        literal = literals.get(column, "")
        if marker != "?":
            unapproved = values.eq("?")
            if unapproved.any():
                row_number = int(unapproved.fillna(False).to_numpy().argmax()) + 1
                raise ValueError(f"Unapproved '?' marker in column {column!r}, data row {row_number}.")
        changed = marker_after = 0
        if marker:
            matches = values.eq(marker)
            changed = int(matches.sum())
            cleaned[column] = values.mask(matches, "")
            marker_after = int(cleaned[column].eq(marker).sum())

        empty_before = int(values.eq("").sum())
        empty_after = int(cleaned[column].eq("").sum())
        audit.append({
            "column": column,
            "rows_before": count,
            "rows_after": len(cleaned),
            "source_marker": marker,
            "marker_before": changed,
            "marker_after": marker_after,
            "empty_before": empty_before,
            "empty_after": empty_after,
            "missing_rate_before": (empty_before + changed) / count if count else None,
            "missing_rate_after": empty_after / count if count else None,
            "changed_cells": changed,
            "preserved_literal": literal,
            "literal_before": int(values.eq(literal).sum()) if literal else 0,
            "literal_after": int(cleaned[column].eq(literal).sum()) if literal else 0,
        })
    return CleaningResult(frame=cleaned, audit=pd.DataFrame(audit, columns=AUDIT_COLUMNS))


def save_cleaning_result(
    result: CleaningResult,
    *,
    cleaned_path: Path = CLEANED_DATA_PATH,
    audit_path: Path = CLEANING_AUDIT_PATH,
) -> tuple[Path, Path]:
    """Write the cleaned dataset and audit with the project's CSV conventions."""
    write_csv(cleaned_path, result.frame)
    write_csv(audit_path, result.audit)
    return cleaned_path, audit_path
