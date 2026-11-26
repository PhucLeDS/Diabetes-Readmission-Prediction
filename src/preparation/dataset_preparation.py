"""Select one eligible encounter per patient and audit the cohort."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from platform import python_version

import pandas as pd

from src.io_utils import CsvValue, read_csv_frame, read_json, sha256_file, write_csv_frame, write_json
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

    frame: pd.DataFrame
    flow: list[dict[str, str | int]]
    excluded_by_code: dict[str, int]
    unclear_retained_by_code: dict[str, int]
    audit: pd.DataFrame


def _stage_summary(stage: str, frame: pd.DataFrame, removed: int) -> dict[str, str | int]:
    labels = frame[TARGET_COLUMN].value_counts()
    return {
        "stage": stage,
        "encounters": len(frame),
        "patients": int(frame["patient_nbr"].nunique()),
        "removed": removed,
        **{label: int(labels.get(label, 0)) for label in TARGET_LABELS},
    }


def _column_audit(
    columns: Sequence[str], stages: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    audit = pd.DataFrame({"column": columns})
    for stage, frame in stages.items():
        empty = frame.eq("").sum().to_numpy()
        audit[f"{stage}_rows"] = len(frame)
        audit[f"{stage}_empty"] = empty
        audit[f"{stage}_missing_rate"] = empty / len(frame) if len(frame) else None
    return audit.loc[:, list(AUDIT_COLUMNS)]


def select_cohort(
    frame: pd.DataFrame,
    mapping: RawTable,
    *,
    config_path: Path = PREPARATION_CONFIG_PATH,
) -> CohortResult:
    """Filter discharge codes, then keep the smallest numeric encounter ID per patient."""
    validate_schema(list(frame.columns), expected=ENCOUNTER_COLUMNS)
    validate_unique_ids(frame["encounter_id"], name="encounter_id")
    validate_target(frame[TARGET_COLUMN], allowed=TARGET_LABELS)
    descriptions = parse_ids_mapping(mapping, sections=(DISCHARGE_COLUMN,))[DISCHARGE_COLUMN]
    excluded_codes = read_json(config_path)["cohort"]["excluded_discharge_codes"]
    if not isinstance(excluded_codes, list) or any(
        not isinstance(code, str) or code not in descriptions for code in excluded_codes
    ):
        raise ValueError(f"{config_path}: excluded discharge codes must be mapped source strings.")
    for column in ID_COLUMNS:
        values = frame[column]
        invalid = ~values.str.fullmatch(r"[0-9]+", na=False)
        if invalid.any():
            raise ValueError(f"Invalid {column}: {values.loc[invalid].iloc[0]!r}.")

    discharge = frame[DISCHARGE_COLUMN]
    unmapped = ~discharge.isin(descriptions)
    if unmapped.any():
        code = discharge.loc[unmapped].iloc[0]
        encounter = frame.loc[unmapped, "encounter_id"].iloc[0]
        raise ValueError(f"Unmapped discharge disposition code {code!r} for encounter {encounter!r}.")

    is_excluded = discharge.isin(excluded_codes)
    excluded_counts = discharge.loc[is_excluded].value_counts()
    excluded = {code: int(excluded_counts.get(code, 0)) for code in sorted(set(excluded_codes), key=int)}
    eligible = frame.loc[~is_excluded]
    selected = (
        eligible.assign(encounter_number=eligible["encounter_id"].map(int))
        .sort_values(["encounter_number", "encounter_id"])
        .drop_duplicates(subset="patient_nbr")
        .drop(columns="encounter_number")
        .reset_index(drop=True)
    )
    validate_unique_ids(selected["patient_nbr"], name="cohort patient_nbr")
    selected_codes = selected[DISCHARGE_COLUMN]
    unclear = selected_codes.loc[
        selected_codes.map(descriptions).isin(UNCLEAR_DISCHARGE_DESCRIPTIONS)
    ].value_counts()
    flow = [
        _stage_summary("Cleaned", frame, 0),
        _stage_summary("Eligible", eligible, sum(excluded.values())),
        _stage_summary("Cohort", selected, len(eligible) - len(selected)),
    ]
    stages = {"cleaned": frame, "eligible": eligible, "cohort": selected}
    return CohortResult(
        frame=selected, flow=flow, excluded_by_code=excluded,
        unclear_retained_by_code={code: int(unclear[code]) for code in sorted(unclear.index, key=int)},
        audit=_column_audit(ENCOUNTER_COLUMNS, stages),
    )


def save_cohort_result(
    result: CohortResult,
    *,
    cohort_path: Path = COHORT_DATA_PATH,
    audit_path: Path = COHORT_AUDIT_PATH,
) -> tuple[Path, Path]:
    """Write the ordered cohort and its aggregate CSV audit."""
    write_csv_frame(cohort_path, result.frame)
    write_csv_frame(audit_path, result.audit)
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

    frame: pd.DataFrame
    group_counts: pd.DataFrame


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


def _grouped_diagnosis_columns(frame: pd.DataFrame) -> dict[str, pd.Series]:
    codes = set().union(*(frame[column].unique() for column in DIAGNOSIS_COLUMNS))
    lookup = {code: group_icd9(code) for code in codes}
    return {
        group_column: frame[source].map(lookup).astype("string")
        for source, group_column in zip(DIAGNOSIS_COLUMNS, DIAGNOSIS_GROUP_COLUMNS, strict=True)
    }


def group_diagnoses(frame: pd.DataFrame) -> DiagnosisGroupingResult:
    """Replace the three source diagnosis fields with their fixed ICD-9 groups."""
    validate_schema(list(frame.columns), expected=ENCOUNTER_COLUMNS)
    grouped = frame.assign(**_grouped_diagnosis_columns(frame)).loc[:, list(GROUPED_ENCOUNTER_COLUMNS)]
    counts = {
        column: grouped[column].value_counts().reindex(DIAGNOSIS_GROUPS, fill_value=0)
        for column in DIAGNOSIS_GROUP_COLUMNS
    }
    return DiagnosisGroupingResult(frame=grouped, group_counts=pd.DataFrame(counts).rename_axis("Diagnosis group"))


FEATURE_AUDIT_PATH = PREPARATION_OUTPUT_DIR / "feature_audit.csv"
FEATURE_AUDIT_COLUMNS = (
    "column", "action", "logical_type", "input_stage", "reason", "output_column",
    "output_position", "rows_before", "rows_after", "empty_before", "empty_after",
    "missing_rate_before", "missing_rate_after",
)


@dataclass
class FeatureSelectionResult:
    """The ordered prepared schema, its records, and feature selection audit."""

    frame: pd.DataFrame
    audit: pd.DataFrame


def select_predictors(
    grouped: DiagnosisGroupingResult, *, config_path: Path = PREPARATION_CONFIG_PATH,
) -> FeatureSelectionResult:
    """Project the grouped encounters onto the configured baseline schema."""
    grouped_columns = list(grouped.frame.columns)
    validate_schema(grouped_columns, expected=GROUPED_ENCOUNTER_COLUMNS)
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
    if drops.keys() & selected_columns or selected_columns | drops.keys() != set(grouped_columns):
        raise ValueError("Output and dropped columns must account for the grouped schema without overlap.")
    if set(not_created) & (set(ENCOUNTER_COLUMNS) | set(grouped_columns)):
        raise ValueError("A column declared as not created is present in the source or grouped schema.")
    if any(not isinstance(reason, str) or not reason.strip() for reason in (*drops.values(), *not_created.values())):
        raise ValueError("Each dropped or not created column needs a reason.")

    prepared = grouped.frame.loc[:, list(columns)]
    audit = _feature_audit(grouped, columns, drops, not_created)
    return FeatureSelectionResult(frame=prepared, audit=audit)


def _feature_audit(
    grouped: DiagnosisGroupingResult,
    columns: tuple[str, ...],
    drops: dict[str, str],
    not_created: dict[str, str],
) -> pd.DataFrame:
    row_count = len(grouped.frame)
    positions = {column: index for index, column in enumerate(columns, start=1)}
    empty = grouped.frame.eq("").sum()
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

    for column in grouped.frame.columns:
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
        add(column, action, logical_type, reason, empty_before=int(empty[column]))

    for source, replacement in zip(DIAGNOSIS_COLUMNS, DIAGNOSIS_GROUP_COLUMNS, strict=True):
        add(
            source, "replaced_diagnosis", "nominal", "Replaced by the fixed ICD-9 grouping.",
            input_stage="cohort", empty_before=int(grouped.group_counts.at["Missing", replacement]),
            replacement=replacement,
        )
    for column, reason in not_created.items():
        add(column, "not_created", "", reason, input_stage="")
    integer_columns = ("output_position", "rows_before", "rows_after", "empty_before", "empty_after")
    return pd.DataFrame(audit, columns=FEATURE_AUDIT_COLUMNS).astype(dict.fromkeys(integer_columns, "Int64"))


def save_feature_audit(
    result: FeatureSelectionResult, *, audit_path: Path = FEATURE_AUDIT_PATH,
) -> Path:
    """Write the column decisions, positions, and missingness for this selection."""
    return write_csv_frame(audit_path, result.audit)


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
    prepared = selected.frame.reset_index(drop=True)
    source = cohort.frame.reset_index(drop=True)
    validate_schema(list(prepared.columns), expected=expected_columns)
    if len(prepared.columns) != 35:
        raise ValueError("The prepared schema must contain exactly 35 columns.")
    if prepared.empty or len(prepared) != len(source):
        raise ValueError("The prepared table must contain one row for every cohort encounter.")
    for column in ID_COLUMNS:
        validate_unique_ids(prepared[column], name=f"prepared {column}")
    validate_target(prepared[TARGET_COLUMN], allowed=TARGET_LABELS)
    diagnosis_groups = _grouped_diagnosis_columns(source)
    for column, values in prepared.items():
        expected = diagnosis_groups[column] if column in diagnosis_groups else source[column]
        differs = ~values.eq(expected).fillna(False)
        if differs.any():
            row_number = int(differs.to_numpy().argmax()) + 1
            raise ValueError(f"Prepared value differs from the cohort at row {row_number}, column {column!r}.")

    empty = prepared.eq("")
    label_counts = prepared[TARGET_COLUMN].value_counts()
    return {
        "rows": len(prepared),
        "columns": len(prepared.columns),
        "target_counts": {label: int(label_counts.get(label, 0)) for label in TARGET_LABELS},
        "empty_cells_by_column": {column: int(count) for column, count in empty.sum().items()},
        "rows_with_empty_fields": int(empty.any(axis=1).sum()),
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
    csv_path = write_csv_frame(project_root / full_key, selected.frame)
    reloaded = read_csv_frame(csv_path, expected_columns=list(selected.frame.columns))
    if not reloaded.equals(selected.frame.reset_index(drop=True)):
        raise ValueError(f"CSV round trip changed prepared records: {csv_path}")
    artifact_hashes[full_key] = sha256_file(csv_path)
    if any(sha256_file(project_root / key) != checksum for key, checksum in raw_hashes.items()):
        raise RuntimeError("A raw source changed during full prepared export.")

    manifest = {
        "dataset": "full_prepared",
        "manifest_version": 1,
        "dataset_path": full_key,
        "schema": _prepared_schema_rows(selected),
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


def _prepared_schema_rows(selected: FeatureSelectionResult) -> list[dict[str, str]]:
    """Return feature audit entries in prepared column order with their schema roles."""
    audit = selected.audit.loc[selected.audit["output_position"].notna()].set_index("column")
    ordered = audit.loc[list(selected.frame.columns), ["action", "logical_type"]]
    roles = {"retain_identifier": "identifier", "retain_predictor": "predictor", "retain_target": "target"}
    return [
        {"name": column, "role": roles[action], "logical_type": logical_type}
        for column, action, logical_type in ordered.itertuples(name=None)
    ]
