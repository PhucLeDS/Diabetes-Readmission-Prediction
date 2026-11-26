"""Load the verified cohort and build the context tables used by EDA."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.io_utils import read_csv, read_json, sha256_file
from src.paths import IDS_MAPPING_PATH, PROJECT_ROOT
from src.preparation.data_loading import COUNT_COLUMNS, ID_COLUMNS, TARGET_COLUMN, load_ids_mapping, parse_ids_mapping
from src.preparation.dataset_export import load_full_prepared, parse_count_columns
from src.preparation.dataset_preparation import (
    FEATURE_AUDIT_PATH,
    PREPARATION_MANIFEST_PATH,
    UNCLEAR_DISCHARGE_DESCRIPTIONS,
)

ADMISSION_COLUMNS = ("admission_type_id", "admission_source_id")
MEDICATION_COLUMNS = (
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "acarbose",
    "insulin", "glyburide-metformin",
)
ADMISSION = "At admission"
CURRENT_ENCOUNTER = "During the representative encounter"
PRIOR_YEAR = "Year before the representative encounter"
UNVERIFIED_AVAILABILITY = "Availability at the intended prediction point is unverified."

# Definitions follow the source descriptions in Strack et al. (2014), Table 1.
# Availability in a real system is distinct from the period a field describes.
COLUMN_DETAILS = {
    "encounter_id": ("Encounter identifier", "Encounter record", "Identifier", "Used for integrity checks."),
    "patient_nbr": ("Patient identifier", "Patient record", "Identifier", "One row per patient in this cohort."),
    "race": ("Recorded race category", "Patient record", "Source category", "An empty field means the category is unavailable."),
    "gender": ("Recorded gender category", "Patient record", "Source category", "Source Unknown/Invalid was converted to an empty field."),
    "age": ("Recorded age band", "Patient record", "Ten-year bands", "The dataset provides age ranges."),
    "admission_type_id": ("Type of admission", ADMISSION, "Source code", "See the Admission Type lookup."),
    "admission_source_id": ("Origin of admission", ADMISSION, "Source code", "See the Admission Source lookup."),
    "time_in_hospital": ("Length of stay from admission to discharge", CURRENT_ENCOUNTER, "Days", UNVERIFIED_AVAILABILITY),
    "num_lab_procedures": ("Laboratory tests performed", CURRENT_ENCOUNTER, "Test count", UNVERIFIED_AVAILABILITY),
    "num_procedures": ("Procedures excluding laboratory tests", CURRENT_ENCOUNTER, "Procedure count", UNVERIFIED_AVAILABILITY),
    "num_medications": ("Distinct medications administered, counted by generic name", CURRENT_ENCOUNTER, "Medication count", UNVERIFIED_AVAILABILITY),
    "number_outpatient": ("Recorded outpatient visits", PRIOR_YEAR, "Visit count", "Zero is a recorded count."),
    "number_emergency": ("Recorded emergency visits", PRIOR_YEAR, "Visit count", "Zero is a recorded count."),
    "number_inpatient": ("Recorded inpatient visits", PRIOR_YEAR, "Visit count", "Zero is a recorded count."),
    "number_diagnoses": ("Diagnoses entered in the source system", CURRENT_ENCOUNTER, "Diagnosis count", UNVERIFIED_AVAILABILITY),
    "max_glu_serum": ("Recorded serum glucose result category", CURRENT_ENCOUNTER, "Source result category", "None means no result is recorded."),
    "A1Cresult": ("Recorded HbA1c result category", CURRENT_ENCOUNTER, "Source result category", "None means no result is recorded."),
    "change": ("Recorded change in diabetes medication", CURRENT_ENCOUNTER, "Ch or No", "Source summary field. " + UNVERIFIED_AVAILABILITY),
    "diabetesMed": ("Whether a diabetes medication was prescribed", CURRENT_ENCOUNTER, "Yes or No", "Source summary field. " + UNVERIFIED_AVAILABILITY),
    "readmitted": ("Recorded readmission status", "After discharge", "NO, <30, or >30", "The source provides categories and records no exact interval."),
    **{
        drug: (
            f"Recorded prescription status for {drug}",
            CURRENT_ENCOUNTER,
            "No, Steady, Up, or Down",
            "The observed levels can differ by medication. " + UNVERIFIED_AVAILABILITY,
        )
        for drug in MEDICATION_COLUMNS
    },
    **{
        f"diag_{position}_group": (
            f"ICD-9 group for the {description} diagnosis",
            CURRENT_ENCOUNTER,
            "Fixed ICD-9 group",
            "Missing indicates an empty source code. Unmapped indicates an unsupported code.",
        )
        for position, description in ((1, "primary"), (2, "secondary"), (3, "additional secondary"))
    },
}


@dataclass
class EdaInputs:
    """The verified cohort and its context tables in prepared column order."""

    cohort: pd.DataFrame
    cohort_flow: pd.DataFrame
    column_dictionary: pd.DataFrame
    source_feature_decisions: pd.DataFrame
    admission_descriptions: pd.DataFrame
    unclear_discharge_count: int


def _admission_descriptions(cohort: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    descriptions = parse_ids_mapping(load_ids_mapping(mapping_path), sections=ADMISSION_COLUMNS)
    records = []
    for column, lookup in descriptions.items():
        observed_codes = set(cohort[column].unique()) - {""}
        for code in sorted(observed_codes | lookup.keys(), key=lambda value: (len(value), value)):
            description = lookup.get(code, "")
            if code not in lookup:
                status = "Unmapped code"
            elif description in UNCLEAR_DISCHARGE_DESCRIPTIONS:
                status = "Unclear description"
            else:
                status = "Mapped"
            records.append({
                "column": column,
                "code": code,
                "description": description,
                "lookup_status": status,
                "observed": code in observed_codes,
            })
    return pd.DataFrame.from_records(records)


def _column_dictionary(schema: list[dict[str, str]]) -> pd.DataFrame:
    if {entry["name"] for entry in schema} != COLUMN_DETAILS.keys():
        raise ValueError("Prepared columns and EDA dictionary entries differ.")
    records = []
    for entry in schema:
        column = entry["name"]
        if column in ID_COLUMNS:
            expected = ("identifier", "identifier")
        elif column == TARGET_COLUMN:
            expected = ("target", "target")
        else:
            expected = ("predictor", "numeric" if column in COUNT_COLUMNS else "nominal")
        if (entry["role"], entry["logical_type"]) != expected:
            raise ValueError(f"Unexpected schema role or logical type for {column!r}.")
        meaning, period, unit_or_domain, notes = COLUMN_DETAILS[column]
        source = column.removesuffix("_group")
        records.append({
            "column": column,
            "role": entry["role"],
            "logical_type": entry["logical_type"],
            "source_column": source,
            "meaning": meaning,
            "period": period,
            "unit_or_domain": unit_or_domain,
            "notes": notes,
        })
    return pd.DataFrame.from_records(records)


def load_eda_inputs(*, project_root: Path = PROJECT_ROOT) -> EdaInputs:
    """Load and describe the full cohort without writing any artifacts."""
    cohort = parse_count_columns(load_full_prepared(project_root=project_root))
    manifest = read_json(project_root / PREPARATION_MANIFEST_PATH.relative_to(PROJECT_ROOT))

    cohort_flow = pd.DataFrame.from_records(manifest["cohort"]["flow"])
    cohort_flow["stage"] = cohort_flow["stage"].replace({"Cleaned": "Cleaned (same rows as raw)"})
    final_stage = cohort_flow.iloc[-1]
    if final_stage["stage"] != "Cohort" or (
        final_stage["encounters"], final_stage["patients"]
    ) != (len(cohort), cohort[ID_COLUMNS[1]].nunique()):
        raise ValueError("The preparation manifest cohort flow differs from the prepared cohort.")

    audit_key = FEATURE_AUDIT_PATH.relative_to(PROJECT_ROOT).as_posix()
    audit_path = project_root / audit_key
    if sha256_file(audit_path) != manifest["artifact_sha256"][audit_key]:
        raise ValueError("The feature audit differs from the preparation manifest.")
    audit = read_csv(audit_path)
    decisions = audit.loc[
        audit["action"].isin(["drop_source", "replaced_diagnosis", "not_created"]),
        ["column", "action", "input_stage", "output_column", "reason"],
    ].reset_index(drop=True)

    return EdaInputs(
        cohort=cohort,
        cohort_flow=cohort_flow,
        column_dictionary=_column_dictionary(manifest["schema"]),
        source_feature_decisions=decisions,
        admission_descriptions=_admission_descriptions(cohort, project_root / IDS_MAPPING_PATH.relative_to(PROJECT_ROOT)),
        unclear_discharge_count=sum(manifest["cohort"]["unclear_discharge_retained"].values()),
    )
