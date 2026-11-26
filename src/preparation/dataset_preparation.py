"""Select one eligible encounter per patient and audit the cohort."""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from platform import python_version

from src.io_utils import CsvValue, read_csv, read_json, sha256_file, write_csv, write_json
from src.paths import (
    CONFIG_DIR, IDS_MAPPING_PATH, INTERIM_DIR, PREPARATION_OUTPUT_DIR,
    PROCESSED_DIR, PROJECT_ROOT, RAW_DATA_PATH,
)
from src.preparation.data_cleaning import CLEANED_DATA_PATH, CLEANING_AUDIT_PATH
from src.preparation.data_loading import (
    COUNT_COLUMNS, ENCOUNTER_COLUMNS, ENCOUNTER_SHA256, ID_COLUMNS,
    MAPPING_SHA256, TARGET_COLUMN, TARGET_LABELS, RawTable, parse_ids_mapping,
)
from src.validation import validate_required_columns, validate_schema, validate_target, validate_unique_ids


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
    descriptions = parse_ids_mapping(mapping, sections=(DISCHARGE_COLUMN,))[DISCHARGE_COLUMN]
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


DIAGNOSIS_COLUMNS = ("diag_1", "diag_2", "diag_3")
DIAGNOSIS_GROUP_COLUMNS = tuple(f"{column}_group" for column in DIAGNOSIS_COLUMNS)
GROUPED_ENCOUNTER_COLUMNS = tuple(
    f"{column}_group" if column in DIAGNOSIS_COLUMNS else column for column in ENCOUNTER_COLUMNS
)
ICD9_GROUP_RULE_VERSION = "icd9_fixed_v1"
ICD9_NUMERIC_GROUPS = (
    ("Diabetes", 250, 250, ()),
    ("Circulatory", 390, 459, (785,)),
    ("Respiratory", 460, 519, (786,)),
    ("Digestive", 520, 579, (787,)),
    ("Genitourinary", 580, 629, (788,)),
    ("Musculoskeletal", 710, 739, ()),
    ("Injury", 800, 999, ()),
    ("Neoplasms", 140, 239, ()),
)
DIAGNOSIS_GROUPS = (
    *(name for name, _, _, _ in ICD9_NUMERIC_GROUPS),
    "Other_numeric", "Other_V", "Other_E", "Missing", "Unmapped",
)
NUMERIC_ICD9 = re.compile(r"[0-9]{1,3}(?:\.[0-9]{1,2})?")
SUPPLEMENTARY_ICD9 = re.compile(r"[VE][0-9]{1,4}(?:\.[0-9]{1,2})?")


@dataclass
class DiagnosisGroupingResult:
    """Records with grouped diagnoses and counts for each diagnosis position."""

    columns: tuple[str, ...]
    rows: list[dict[str, str]]
    group_counts: dict[str, dict[str, int]]


def group_icd9(value: str) -> str:
    """Apply the fixed research grouping to an ICD-9 field from the cleaned cohort."""
    if value == "":
        return "Missing"

    code = value.strip().upper()
    if SUPPLEMENTARY_ICD9.fullmatch(code):
        return f"Other_{code[0]}"
    if not NUMERIC_ICD9.fullmatch(code):
        return "Unmapped"

    number = int(code.split(".", 1)[0])
    for group, first, last, extra_codes in ICD9_NUMERIC_GROUPS:
        if first <= number <= last or number in extra_codes:
            return group
    return "Other_numeric"


def group_diagnoses(
    columns: Sequence[str], rows: list[dict[str, str]],
) -> DiagnosisGroupingResult:
    """Replace the three source diagnosis columns in place within each row's schema."""
    validate_schema(columns, expected=ENCOUNTER_COLUMNS)
    output_columns = GROUPED_ENCOUNTER_COLUMNS
    counts = {column: Counter() for column in DIAGNOSIS_GROUP_COLUMNS}
    grouped_rows = []
    for source in rows:
        row = dict(zip(output_columns, (source[column] for column in columns), strict=True))
        for column, group_counts in counts.items():
            group = row[column] = group_icd9(row[column])
            group_counts[group] += 1
        grouped_rows.append(row)

    return DiagnosisGroupingResult(
        columns=output_columns,
        rows=grouped_rows,
        group_counts={
            column: {group: column_counts[group] for group in DIAGNOSIS_GROUPS}
            for column, column_counts in counts.items()
        },
    )


FEATURE_AUDIT_PATH = PREPARATION_OUTPUT_DIR / "feature_audit.csv"
FEATURE_AUDIT_COLUMNS = (
    "column", "action", "logical_type", "input_stage", "reason", "output_column",
    "output_position", "rows_before", "rows_after", "empty_before", "empty_after",
    "missing_rate_before", "missing_rate_after",
)


@dataclass
class FeatureSelectionResult:
    """The ordered prepared schema, its records, and feature selection audit."""

    columns: tuple[str, ...]
    rows: list[dict[str, str]]
    audit: list[dict[str, CsvValue]]


def select_predictors(
    grouped: DiagnosisGroupingResult, *, config_path: Path = PREPARATION_CONFIG_PATH,
) -> FeatureSelectionResult:
    """Project the grouped encounters onto the configured baseline schema."""
    validate_schema(grouped.columns, expected=GROUPED_ENCOUNTER_COLUMNS)
    policy = read_json(config_path)["feature_selection"]
    predictors = policy["predictors"]
    drops = policy["dropped_source_columns"]
    not_created = policy["not_created_columns"]
    if not isinstance(predictors, list) or len(predictors) != 32 or not all(
        isinstance(name, str) for name in predictors
    ):
        raise ValueError("Expected an ordered list of 32 predictor names.")
    columns = (*ID_COLUMNS, *predictors, TARGET_COLUMN)
    validate_schema(columns)
    validate_required_columns(predictors, DIAGNOSIS_GROUP_COLUMNS)
    if not isinstance(drops, dict) or not isinstance(not_created, dict):
        raise ValueError("Expected name-to-reason mappings for dropped and uncreated columns.")
    selected_columns = set(columns)
    if drops.keys() & selected_columns or selected_columns | drops.keys() != set(grouped.columns):
        raise ValueError("Output and dropped columns must account for the grouped schema without overlap.")
    if set(not_created) & (set(ENCOUNTER_COLUMNS) | set(grouped.columns)):
        raise ValueError("A column declared as not created is present in the source or grouped schema.")
    if any(not isinstance(reason, str) or not reason.strip() for reason in (*drops.values(), *not_created.values())):
        raise ValueError("Each dropped or not created column needs a reason.")

    rows = [{column: row[column] for column in columns} for row in grouped.rows]
    audit = _feature_audit(grouped, columns, drops, not_created)
    return FeatureSelectionResult(columns=columns, rows=rows, audit=audit)


def _feature_audit(
    grouped: DiagnosisGroupingResult,
    columns: tuple[str, ...],
    drops: dict[str, str],
    not_created: dict[str, str],
) -> list[dict[str, CsvValue]]:
    row_count = len(grouped.rows)
    positions = {column: index for index, column in enumerate(columns, start=1)}
    empty = Counter(column for row in grouped.rows for column, value in row.items() if value == "")
    audit: list[dict[str, CsvValue]] = []

    def add(
        column: str, action: str, logical_type: str, reason: str, *,
        input_stage: str = "grouped_cohort", empty_before: int | None = None,
        replacement: str = "",
    ) -> None:
        output_position = positions.get(column)
        present = output_position is not None
        before_rate = empty_before / row_count if row_count and empty_before is not None else None
        audit.append({
            "column": column,
            "action": action,
            "logical_type": logical_type,
            "input_stage": input_stage,
            "reason": reason,
            "output_column": column if present else replacement,
            "output_position": output_position,
            "rows_before": row_count if input_stage else None,
            "rows_after": row_count if present else None,
            "empty_before": empty_before,
            "empty_after": empty_before if present else None,
            "missing_rate_before": before_rate,
            "missing_rate_after": before_rate if present else None,
        })

    for column in grouped.columns:
        if column in ID_COLUMNS:
            action, logical_type, reason = "retain_identifier", "identifier", "Integrity checks and split tracking."
        elif column == TARGET_COLUMN:
            action, logical_type, reason = "retain_target", "target", "Preserve the original three labels."
        elif column in drops:
            action, logical_type, reason = "drop_source", "nominal", drops[column]
        else:
            action = "retain_predictor"
            logical_type = "numeric" if column in COUNT_COLUMNS else "nominal"
            reason = "Included in the approved baseline."
        add(column, action, logical_type, reason, empty_before=empty[column])

    for source, replacement in zip(DIAGNOSIS_COLUMNS, DIAGNOSIS_GROUP_COLUMNS, strict=True):
        add(
            source, "replaced_diagnosis", "nominal", "Replaced by the fixed ICD-9 grouping.",
            input_stage="cohort", empty_before=grouped.group_counts[replacement]["Missing"],
            replacement=replacement,
        )
    for column, reason in not_created.items():
        add(column, "not_created", "", reason, input_stage="")
    return audit


def save_feature_audit(
    result: FeatureSelectionResult, *, audit_path: Path = FEATURE_AUDIT_PATH,
) -> Path:
    """Write the column decisions, positions, and missingness for this selection."""
    return write_csv(audit_path, FEATURE_AUDIT_COLUMNS, result.audit)


FULL_PREPARED_PATH = PROCESSED_DIR / "full_prepared.csv"
PREPARATION_MANIFEST_PATH = PREPARATION_OUTPUT_DIR / "preparation_manifest.json"


def preparation_config_checksums(path: Path) -> tuple[str, tuple[str, ...]]:
    """Return the LF checksum and the accepted checksums from earlier exports."""
    content = path.read_bytes()
    lf_content = content.replace(b"\r\n", b"\n")
    canonical = sha256(lf_content).hexdigest()
    compatible = (
        canonical,
        sha256(lf_content.replace(b"\n", b"\r\n")).hexdigest(),
        sha256(content).hexdigest(),
    )
    return canonical, compatible


def validate_full_prepared(
    selected: FeatureSelectionResult,
    cohort: CohortResult,
    *,
    config_path: Path = PREPARATION_CONFIG_PATH,
) -> dict:
    """Verify schema, identifiers, and values against the ordered cohort."""
    predictors = read_json(config_path)["feature_selection"]["predictors"]
    expected_columns = (*ID_COLUMNS, *predictors, TARGET_COLUMN)
    validate_schema(selected.columns, expected=expected_columns)
    if len(selected.columns) != 35:
        raise ValueError("The prepared schema must contain exactly 35 columns.")
    if not selected.rows or len(selected.rows) != len(cohort.rows):
        raise ValueError("The prepared table must contain one row for every cohort encounter.")
    for column in ID_COLUMNS:
        validate_unique_ids((row[column] for row in selected.rows), name=f"prepared {column}")
    validate_target((row[TARGET_COLUMN] for row in selected.rows), allowed=TARGET_LABELS)
    diagnosis_sources = dict(zip(DIAGNOSIS_GROUP_COLUMNS, DIAGNOSIS_COLUMNS, strict=True))

    empty_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    rows_with_empty = 0
    for row_number, (source, prepared) in enumerate(zip(cohort.rows, selected.rows, strict=True), start=1):
        if tuple(prepared) != selected.columns:
            raise ValueError(f"Unexpected prepared record schema at row {row_number}.")
        has_empty = False
        for column, value in prepared.items():
            if column in diagnosis_sources:
                expected = group_icd9(source[diagnosis_sources[column]])
            else:
                expected = source[column]
            if value != expected:
                raise ValueError(f"Prepared value differs from the cohort at row {row_number}, column {column!r}.")
            if value == "":
                empty_counts[column] += 1
                has_empty = True
        rows_with_empty += has_empty
        label_counts[prepared[TARGET_COLUMN]] += 1

    return {
        "rows": len(selected.rows),
        "columns": len(selected.columns),
        "target_counts": {label: label_counts[label] for label in TARGET_LABELS},
        "empty_cells_by_column": {column: empty_counts[column] for column in selected.columns},
        "rows_with_empty_fields": rows_with_empty,
    }


def save_full_prepared(
    selected: FeatureSelectionResult, cohort: CohortResult, *, project_root: Path = PROJECT_ROOT,
) -> tuple[Path, Path, dict]:
    """Save the prepared CSV, verify the round trip, and write its manifest."""
    config_key = PREPARATION_CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix()
    config_path = project_root / config_key
    statistics = validate_full_prepared(selected, cohort, config_path=config_path)
    config_checksum, compatible_checksums = preparation_config_checksums(config_path)
    raw_hashes = {}
    for source, expected in ((RAW_DATA_PATH, ENCOUNTER_SHA256), (IDS_MAPPING_PATH, MAPPING_SHA256)):
        key = source.relative_to(PROJECT_ROOT).as_posix()
        raw_hashes[key] = sha256_file(project_root / key)
        if raw_hashes[key] != expected:
            raise ValueError(f"Unexpected raw checksum for {key}.")

    artifact_paths = (
        CLEANED_DATA_PATH, COHORT_DATA_PATH, CLEANING_AUDIT_PATH,
        COHORT_AUDIT_PATH, FEATURE_AUDIT_PATH,
    )
    artifact_keys = [path.relative_to(PROJECT_ROOT).as_posix() for path in artifact_paths]
    artifact_hashes = {key: sha256_file(project_root / key) for key in artifact_keys}
    full_key = FULL_PREPARED_PATH.relative_to(PROJECT_ROOT).as_posix()
    csv_path = write_csv(project_root / full_key, selected.columns, selected.rows)
    _, reloaded = read_csv(csv_path, expected_columns=selected.columns)
    if reloaded != selected.rows:
        raise ValueError(f"CSV round trip changed prepared records: {csv_path}")
    artifact_hashes[full_key] = sha256_file(csv_path)
    if any(sha256_file(project_root / key) != checksum for key, checksum in raw_hashes.items()):
        raise RuntimeError("A raw source changed during full prepared export.")

    schema = [
        {
            "name": row["column"],
            "role": role,
            "logical_type": row["logical_type"],
        }
        for row, role in _prepared_schema_rows(selected)
    ]
    manifest = {
        "dataset": "full_prepared",
        "manifest_version": 1,
        "dataset_path": full_key,
        "schema": schema,
        "statistics": statistics,
        "cohort": {
            "flow": cohort.flow,
            "excluded_by_discharge_code": cohort.excluded_by_code,
            "unclear_discharge_retained": cohort.unclear_retained_by_code,
        },
        "raw_sha256": raw_hashes,
        "configuration": {"path": config_key, "sha256": config_checksum},
        "artifact_sha256": artifact_hashes,
        "validation": {
            "ordered_schema": "passed",
            "unique_identifiers": "passed",
            "values_and_labels_against_cohort": "passed",
            "csv_round_trip": "passed",
            "raw_checksums": "passed",
        },
        "runtime": {"python": python_version()},
    }
    manifest_path = project_root / PREPARATION_MANIFEST_PATH.relative_to(PROJECT_ROOT)
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        existing_config = existing.get("configuration") if isinstance(existing, dict) else None
        if isinstance(existing_config, dict) and existing_config.get("sha256") in compatible_checksums:
            # Reuse the earlier digest only when the protected write matches the entire manifest.
            manifest["configuration"]["sha256"] = existing_config["sha256"]
    write_json(manifest_path, manifest)
    return csv_path, manifest_path, manifest


def _prepared_schema_rows(selected: FeatureSelectionResult):
    """Return feature audit entries in prepared column order with their schema roles."""
    audit = {row["column"]: row for row in selected.audit if row["output_position"] is not None}
    roles = {"retain_identifier": "identifier", "retain_predictor": "predictor", "retain_target": "target"}
    return [(audit[column], roles[audit[column]["action"]]) for column in selected.columns]
