"""Select one eligible encounter per patient and audit the cohort."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.io_utils import CsvValue, read_json, write_csv
from src.paths import CONFIG_DIR, INTERIM_DIR, PREPARATION_OUTPUT_DIR
from src.preparation.data_loading import ENCOUNTER_COLUMNS, TARGET_COLUMN, TARGET_LABELS, RawTable
from src.validation import validate_schema, validate_target, validate_unique_ids


COHORT_DATA_PATH = INTERIM_DIR / "cohort.csv"
COHORT_AUDIT_PATH = PREPARATION_OUTPUT_DIR / "cohort_audit.csv"
PREPARATION_CONFIG_PATH = CONFIG_DIR / "preparation.json"
DISCHARGE_COLUMN = "discharge_disposition_id"
UNCLEAR_DISCHARGE_DESCRIPTIONS = {"NULL", "Not Mapped", "Unknown/Invalid", "Not Available"}
AUDIT_COLUMNS = (
    "column", "cleaned_rows", "eligible_rows", "cohort_rows",
    "cleaned_empty", "eligible_empty", "cohort_empty",
    "cleaned_missing_rate", "eligible_missing_rate", "cohort_missing_rate",
)


@dataclass
class CohortResult:
    """Selected records and aggregate evidence for each selection step."""

    columns: tuple[str, ...]
    rows: list[dict[str, str]]
    flow: list[dict[str, str | int]]
    excluded_by_code: dict[str, int]
    unclear_retained_by_code: dict[str, int]
    audit: list[dict[str, CsvValue]]


def _numeric_id(value: str, column: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"Invalid {column}: {value!r}.")
    return int(value)


def _discharge_descriptions(mapping: RawTable) -> dict[str, str]:
    """Read the discharge section for code validation and audit labels."""
    code_column, description_column = mapping.columns
    for index, row in enumerate(mapping.rows):
        if row[code_column] == DISCHARGE_COLUMN and row[description_column] == "description":
            break
    else:
        raise ValueError("The source mapping has no discharge disposition section.")

    descriptions = {}
    for row in mapping.rows[index + 1:]:
        code, description = row[code_column], row[description_column].strip()
        if not code and not description:
            break
        _numeric_id(code, DISCHARGE_COLUMN)
        if not description or code in descriptions:
            raise ValueError(f"Missing or duplicate discharge description for code {code!r}.")
        descriptions[code] = description
    if not descriptions:
        raise ValueError("The discharge disposition section has no codes.")
    return descriptions


def _stage_summary(stage: str, rows: list[dict[str, str]], removed: int) -> dict[str, str | int]:
    labels = Counter(row[TARGET_COLUMN] for row in rows)
    return {
        "stage": stage,
        "encounters": len(rows),
        "patients": len({row["patient_nbr"] for row in rows}),
        "removed": removed,
        **{label: labels[label] for label in TARGET_LABELS},
    }


def _column_audit(
    columns: Sequence[str], stages: dict[str, list[dict[str, str]]],
) -> list[dict[str, CsvValue]]:
    result = []
    for column in columns:
        record: dict[str, CsvValue] = {"column": column}
        for stage, rows in stages.items():
            empty = sum(row[column] == "" for row in rows)
            record[f"{stage}_rows"] = len(rows)
            record[f"{stage}_empty"] = empty
            record[f"{stage}_missing_rate"] = empty / len(rows) if rows else None
        result.append(record)
    return result


def select_cohort(
    columns: Sequence[str],
    rows: list[dict[str, str]],
    mapping: RawTable,
    *,
    config_path: Path = PREPARATION_CONFIG_PATH,
) -> CohortResult:
    """Filter discharge codes, then keep the smallest numeric encounter ID per patient."""
    validate_schema(columns, expected=ENCOUNTER_COLUMNS)
    validate_unique_ids((row["encounter_id"] for row in rows), name="encounter_id")
    validate_target((row[TARGET_COLUMN] for row in rows), allowed=TARGET_LABELS)
    descriptions = _discharge_descriptions(mapping)
    excluded_codes = read_json(config_path)["cohort"]["excluded_discharge_codes"]
    if not isinstance(excluded_codes, list) or any(
        not isinstance(code, str) or code not in descriptions for code in excluded_codes
    ):
        raise ValueError(f"{config_path}: excluded discharge codes must be mapped source strings.")
    excluded = dict.fromkeys(sorted(set(excluded_codes), key=int), 0)

    eligible = []
    selected_by_patient = {}
    for row in rows:
        encounter = row["encounter_id"]
        rank = (_numeric_id(encounter, "encounter_id"), encounter)
        patient = row["patient_nbr"]
        _numeric_id(patient, "patient_nbr")
        code = row[DISCHARGE_COLUMN]
        if code not in descriptions:
            raise ValueError(f"Unmapped discharge disposition code {code!r} for encounter {encounter!r}.")
        if code in excluded:
            excluded[code] += 1
            continue
        eligible.append(row)
        previous = selected_by_patient.get(patient)
        if previous is None or rank < previous[0]:
            selected_by_patient[patient] = (rank, row)

    selected = [row for _, row in sorted(selected_by_patient.values(), key=lambda entry: entry[0])]
    validate_unique_ids((row["patient_nbr"] for row in selected), name="cohort patient_nbr")
    unclear = Counter(
        row[DISCHARGE_COLUMN] for row in selected
        if descriptions[row[DISCHARGE_COLUMN]] in UNCLEAR_DISCHARGE_DESCRIPTIONS
    )
    flow = [
        _stage_summary("Cleaned", rows, 0),
        _stage_summary("Eligible", eligible, sum(excluded.values())),
        _stage_summary("Cohort", selected, len(eligible) - len(selected)),
    ]
    stages = {"cleaned": rows, "eligible": eligible, "cohort": selected}
    return CohortResult(
        columns=tuple(columns), rows=selected, flow=flow, excluded_by_code=excluded,
        unclear_retained_by_code=dict(sorted(unclear.items(), key=lambda entry: int(entry[0]))),
        audit=_column_audit(columns, stages),
    )


def save_cohort_result(
    result: CohortResult,
    *,
    cohort_path: Path = COHORT_DATA_PATH,
    audit_path: Path = COHORT_AUDIT_PATH,
) -> tuple[Path, Path]:
    """Write the ordered cohort and its aggregate CSV audit."""
    write_csv(cohort_path, result.columns, result.rows)
    write_csv(audit_path, AUDIT_COLUMNS, result.audit)
    return cohort_path, audit_path
