"""Load the prepared cohort and construct its binary modeling view."""

from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path

from src.io_utils import read_csv, read_json, sha256_file
from src.paths import PROJECT_ROOT
from src.preparation.data_loading import ID_COLUMNS, TARGET_COLUMN, TARGET_LABELS
from src.preparation.dataset_preparation import (
    FULL_PREPARED_PATH,
    PREPARATION_CONFIG_PATH,
    PREPARATION_MANIFEST_PATH,
)
from src.validation import validate_required_columns, validate_schema, validate_target, validate_unique_ids


TARGET_MAPPING_CONFIG_PATH = PROJECT_ROOT / "config" / "target_mapping.json"
CLASS_COLUMN = "class"
CLASS_LABELS = ("not_readmitted", "readmitted")


def load_full_prepared(*, project_root: Path = PROJECT_ROOT) -> tuple[list[str], list[dict[str, str]]]:
    """Read the verified Phase II CSV and retain its exact field values."""
    data_path = FULL_PREPARED_PATH.relative_to(PROJECT_ROOT)
    config_path = PREPARATION_CONFIG_PATH.relative_to(PROJECT_ROOT)
    manifest = read_json(project_root / PREPARATION_MANIFEST_PATH.relative_to(PROJECT_ROOT))
    csv_path = project_root / data_path
    configuration = manifest["configuration"]
    config_bytes = (project_root / config_path).read_bytes()
    config_lf = config_bytes.replace(b"\r\n", b"\n")
    # Git can convert the tracked configuration between LF and CRLF on checkout.
    config_checksums = {
        sha256(content).hexdigest()
        for content in (config_bytes, config_lf, config_lf.replace(b"\n", b"\r\n"))
    }
    if (
        configuration["path"] != config_path.as_posix()
        or configuration["sha256"] not in config_checksums
    ):
        raise ValueError("Preparation configuration differs from the manifest.")
    if (
        manifest["dataset_path"] != data_path.as_posix()
        or manifest["artifact_sha256"][data_path.as_posix()] != sha256_file(csv_path)
    ):
        raise ValueError("Full prepared CSV differs from the preparation manifest.")

    predictors = read_json(project_root / config_path)["feature_selection"]["predictors"]
    expected_columns = (*ID_COLUMNS, *predictors, TARGET_COLUMN)
    if len(predictors) != 32 or CLASS_COLUMN in predictors:
        raise ValueError("The preparation configuration must declare 32 predictors without the model class.")
    validate_schema([entry["name"] for entry in manifest["schema"]], expected=expected_columns)
    columns, rows = read_csv(csv_path, expected_columns=expected_columns)

    statistics = manifest["statistics"]
    if not rows or (len(rows), len(columns)) != (statistics["rows"], statistics["columns"]):
        raise ValueError("Full prepared CSV dimensions differ from the preparation manifest.")
    for column in ID_COLUMNS:
        validate_unique_ids((row[column] for row in rows), name=f"prepared {column}")
    validate_target((row[TARGET_COLUMN] for row in rows), allowed=TARGET_LABELS)
    counts = Counter(row[TARGET_COLUMN] for row in rows)
    if {label: counts[label] for label in TARGET_LABELS} != statistics["target_counts"]:
        raise ValueError("Original label counts differ from the preparation manifest.")
    return columns, rows


def load_target_mapping(config_path: Path = TARGET_MAPPING_CONFIG_PATH) -> dict[str, str]:
    """Require the configured source labels and their Any Readmission classes."""
    config = read_json(config_path)
    negative_class, positive_class = CLASS_LABELS
    mapping = config["class_mapping"]
    if (
        config["task_definition"] != "any_readmission"
        or config["target_source"] != TARGET_COLUMN
        or config["class_column"] != CLASS_COLUMN
        or config["class_values"] != list(CLASS_LABELS)
        or config["positive_class"] != positive_class
        or set(mapping) != set(TARGET_LABELS)
        or mapping["NO"] != negative_class
        or mapping["<30"] != positive_class
        or mapping[">30"] != positive_class
    ):
        raise ValueError("Target configuration must implement Any Readmission.")
    return mapping


def map_binary_targets(
    rows: Sequence[Mapping[str, str]], *, config_path: Path = TARGET_MAPPING_CONFIG_PATH,
) -> list[dict[str, str]]:
    """Append class to copies of the records while retaining identifiers and source labels."""
    mapping = load_target_mapping(config_path)
    validate_target((row.get(TARGET_COLUMN, "") for row in rows), allowed=mapping)
    if any(CLASS_COLUMN in row for row in rows):
        raise ValueError(f"Input records already contain the {CLASS_COLUMN!r} column.")
    return [{**row, CLASS_COLUMN: mapping[row[TARGET_COLUMN]]} for row in rows]


def build_model_dataset(
    prepared_columns: Sequence[str], mapped_rows: Sequence[Mapping[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """Project onto the ordered predictors followed by the binary class."""
    validate_schema(prepared_columns)
    if (
        len(prepared_columns) != 35
        or tuple(prepared_columns[:2]) != ID_COLUMNS
        or prepared_columns[-1] != TARGET_COLUMN
        or CLASS_COLUMN in prepared_columns
    ):
        raise ValueError("Expected two identifiers, 32 predictors, and the original label in prepared order.")

    model_columns = [*prepared_columns[2:-1], CLASS_COLUMN]
    validate_target((row.get(CLASS_COLUMN, "") for row in mapped_rows), allowed=CLASS_LABELS)
    model_rows = []
    for row in mapped_rows:
        validate_required_columns(row, model_columns)
        model_rows.append({column: row[column] for column in model_columns})
    return model_columns, model_rows
