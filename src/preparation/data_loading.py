"""Load and inspect the unmodified UCI source tables."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.io_utils import read_csv, sha256_file
from src.paths import IDS_MAPPING_PATH, RAW_DATA_PATH
from src.validation import validate_target, validate_unique_ids


ENCOUNTER_COLUMNS = (
    "encounter_id", "patient_nbr", "race", "gender", "age", "weight",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "payer_code", "medical_specialty", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide",
    "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone", "change", "diabetesMed",
    "readmitted",
)
MAPPING_COLUMNS = ("admission_type_id", "description")
MAPPING_SECTIONS = ("admission_type_id", "discharge_disposition_id", "admission_source_id")
ID_COLUMNS = ("encounter_id", "patient_nbr")
COUNT_COLUMNS = (
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
)
TARGET_COLUMN = "readmitted"
TARGET_LABELS = ("NO", "<30", ">30")
RAW_MARKERS = ("", "?", "None", "Unknown/Invalid")

ENCOUNTER_ROWS = 101_766
MAPPING_ROWS = 67
ENCOUNTER_SHA256 = "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97"
MAPPING_SHA256 = "f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e"


@dataclass(frozen=True)
class RawTable:
    """Source file metadata and CSV records in their original order."""

    path: Path
    sha256: str
    size_bytes: int
    columns: list[str]
    rows: list[dict[str, str]]


def _load_verified_csv(
    path: Path, *, expected_columns: tuple[str, ...], expected_rows: int, expected_sha256: str,
) -> RawTable:
    before = path.stat()
    checksum = sha256_file(path)
    if checksum != expected_sha256:
        raise ValueError(f"{path}: SHA-256 mismatch. Expected {expected_sha256}, got {checksum}.")
    columns, rows = read_csv(path, expected_columns=expected_columns)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino, after.st_size, after.st_mtime_ns,
    ):
        raise RuntimeError(f"{path}: source changed while being read.")
    if len(rows) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} data rows, got {len(rows)}.")
    return RawTable(path=path, sha256=checksum, size_bytes=after.st_size, columns=columns, rows=rows)


def load_raw_encounters(path: Path = RAW_DATA_PATH) -> RawTable:
    """Load the canonical encounters and validate IDs, counts, and original labels."""
    table = _load_verified_csv(
        path, expected_columns=ENCOUNTER_COLUMNS, expected_rows=ENCOUNTER_ROWS,
        expected_sha256=ENCOUNTER_SHA256,
    )
    for row_number, row in enumerate(table.rows, start=1):
        for column in (*ID_COLUMNS, *COUNT_COLUMNS):
            value = row[column]
            if not (value.isascii() and value.isdecimal()):
                raise ValueError(
                    f"{path}: expected nonnegative integer text in data row "
                    f"{row_number}, column {column!r}, got {value!r}."
                )
    validate_unique_ids((row["encounter_id"] for row in table.rows), name="encounter_id")
    validate_target((row[TARGET_COLUMN] for row in table.rows), allowed=TARGET_LABELS)
    return table


def load_ids_mapping(path: Path = IDS_MAPPING_PATH) -> RawTable:
    """Verify and read the mapping file with its original sections and separators."""
    return _load_verified_csv(
        path, expected_columns=MAPPING_COLUMNS, expected_rows=MAPPING_ROWS,
        expected_sha256=MAPPING_SHA256,
    )


def parse_ids_mapping(mapping: RawTable, *, sections: Sequence[str]) -> dict[str, dict[str, str]]:
    """Validate and return lookup entries for the requested source sections."""
    code_field, description_field = MAPPING_COLUMNS
    descriptions: dict[str, dict[str, str]] = {section: {} for section in sections}
    current_section = code_field
    for row in mapping.rows:
        code, description = row[code_field], row[description_field].strip()
        if code in MAPPING_SECTIONS and description == description_field:
            current_section = code
            continue
        if current_section not in descriptions or not (code or description):
            continue
        lookup = descriptions[current_section]
        if not (code.isascii() and code.isdecimal() and description):
            raise ValueError(f"Invalid lookup entry for {current_section}: {code!r}.")
        if code in lookup:
            raise ValueError(f"Duplicate lookup code for {current_section}: {code!r}.")
        lookup[code] = description

    for section, lookup in descriptions.items():
        if not lookup:
            raise ValueError(f"No mapping entries found for {section}.")
    return descriptions


def summarize_raw_encounters(table: RawTable) -> dict:
    """Count encounters, patients, and the three original labels."""
    patient_counts = Counter(row["patient_nbr"] for row in table.rows)
    target_counts = Counter(row[TARGET_COLUMN] for row in table.rows)
    return {
        "encounters": len(table.rows),
        "columns": len(table.columns),
        "patients": len(patient_counts),
        "patients_with_multiple_encounters": sum(count > 1 for count in patient_counts.values()),
        "target_counts": {label: target_counts[label] for label in TARGET_LABELS},
    }


def profile_raw_columns(table: RawTable) -> list[dict[str, str | int]]:
    """Show logical column roles and counts of raw markers without normalizing them."""
    logical_types = dict.fromkeys(ID_COLUMNS, "identifier")
    logical_types.update(dict.fromkeys(COUNT_COLUMNS, "integer"))
    logical_types[TARGET_COLUMN] = "target"
    profile = []
    for column in table.columns:
        markers = Counter(row[column] for row in table.rows if row[column] in RAW_MARKERS)
        profile.append({
            "column": column,
            "logical_type": logical_types.get(column, "categorical"),
            **{marker: markers[marker] for marker in RAW_MARKERS},
        })
    return profile
