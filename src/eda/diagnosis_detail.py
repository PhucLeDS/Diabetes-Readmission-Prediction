"""Measure what the three ICD-9 positions contribute to the prepared cohort."""

from dataclasses import dataclass

import pandas as pd

from src.eda.cohort_selection import MINIMUM_GROUP_SIZE
from src.preparation.data_loading import ID_COLUMNS
from src.preparation.dataset_preparation import (
    DIAGNOSIS_COLUMNS, DIAGNOSIS_GROUP_COLUMNS, DIAGNOSIS_GROUPS, group_icd9,
)


MAPPED_GROUPS = tuple(group for group in DIAGNOSIS_GROUPS if group not in {"Missing", "Unmapped"})


@dataclass
class DiagnosisDetail:
    """Counts for diagnosis depth, group coverage, co-occurrence, and source codes."""

    diagnosis_position_depth: pd.DataFrame
    diagnosis_group_gain: pd.DataFrame
    diagnosis_group_pairs: pd.DataFrame
    diagnosis_source_code_summary: pd.DataFrame
    diagnosis_source_code_detail: pd.DataFrame


def _align_diagnoses(cohort: pd.DataFrame, source_cohort: pd.DataFrame) -> pd.DataFrame:
    aligned = cohort.loc[:, [*ID_COLUMNS, *DIAGNOSIS_GROUP_COLUMNS]].merge(
        source_cohort.loc[:, [*ID_COLUMNS, *DIAGNOSIS_COLUMNS]],
        on="encounter_id", how="inner", suffixes=("", "_source"), validate="one_to_one",
    )
    if len(aligned) != len(cohort) or len(aligned) != len(source_cohort):
        raise ValueError("Prepared and source diagnosis encounter IDs differ.")
    if not aligned["patient_nbr"].eq(aligned["patient_nbr_source"]).fillna(False).all():
        raise ValueError("Patient IDs differ between prepared and source diagnoses.")
    return aligned


def _diagnosis_positions(aligned: pd.DataFrame) -> pd.DataFrame:
    positions = pd.concat([
        aligned[["patient_nbr", source, group]].rename(columns={source: "source_code", group: "group"})
        for source, group in zip(DIAGNOSIS_COLUMNS, DIAGNOSIS_GROUP_COLUMNS, strict=True)
    ], ignore_index=True)
    lookup = {code: group_icd9(code) for code in positions["source_code"].unique()}
    if not positions["source_code"].map(lookup).eq(positions["group"]).fillna(False).all():
        raise ValueError("An ICD-9 source code disagrees with its prepared diagnosis group.")
    return positions


def _position_depth(groups: pd.DataFrame) -> pd.DataFrame:
    mapped = groups.where(groups.isin(MAPPED_GROUPS))
    depth = pd.DataFrame({
        "mapped_positions": mapped.count(axis=1),
        "distinct_mapped_groups": mapped.nunique(axis=1),
        "missing_positions": groups.eq("Missing").sum(axis=1),
        "unmapped_positions": groups.eq("Unmapped").sum(axis=1),
    })
    counts = depth.value_counts(sort=False).sort_index().reset_index(name="patients")
    return counts.assign(
        cohort_size=len(groups), percent_of_cohort=100 * counts["patients"] / len(groups),
        small_cell=counts["patients"].lt(MINIMUM_GROUP_SIZE),
    )


def _group_gain(diagnosis_summary: pd.DataFrame) -> pd.DataFrame:
    summary = diagnosis_summary.loc[
        diagnosis_summary["group"].isin(MAPPED_GROUPS),
        ["group", "diag_1_group_patients", "patients_at_any_position", "cohort_size"],
    ].rename(columns={"diag_1_group_patients": "patients_at_primary_position"})
    extra = summary["patients_at_any_position"] - summary["patients_at_primary_position"]
    return summary.assign(
        additional_patients=extra,
        primary_percent_of_cohort=100 * summary["patients_at_primary_position"] / summary["cohort_size"],
        any_position_percent_of_cohort=100 * summary["patients_at_any_position"] / summary["cohort_size"],
        additional_percent_of_cohort=100 * extra / summary["cohort_size"],
        small_primary_cell=summary["patients_at_primary_position"].lt(MINIMUM_GROUP_SIZE),
        small_any_position_cell=summary["patients_at_any_position"].lt(MINIMUM_GROUP_SIZE),
    )


def _group_pairs(positions: pd.DataFrame, cohort_size: int) -> pd.DataFrame:
    groups = positions.loc[positions["group"].isin(MAPPED_GROUPS), ["patient_nbr", "group"]].drop_duplicates()
    pairs = groups.merge(groups, on="patient_nbr", suffixes=("_1", "_2"))
    pairs = pairs.loc[pairs["group_1"].lt(pairs["group_2"]), ["group_1", "group_2"]]
    counts = pairs.value_counts().rename("patients").reset_index()
    return counts.sort_values(
        ["patients", "group_1", "group_2"], ascending=[False, True, True], ignore_index=True,
    ).assign(
        cohort_size=cohort_size, percent_of_cohort=lambda rows: 100 * rows["patients"] / cohort_size,
        small_cell=lambda rows: rows["patients"].lt(MINIMUM_GROUP_SIZE),
    )


def _source_code_counts(positions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coded = positions.loc[positions["source_code"].ne("")].drop_duplicates()
    group_sizes = coded[["patient_nbr", "group"]].drop_duplicates()["group"].value_counts()
    detail = (
        coded.groupby(["group", "source_code"]).size().reset_index(name="patients")
        .sort_values(["group", "patients", "source_code"], ascending=[True, False, True], ignore_index=True)
    )
    detail["group_patients"] = detail["group"].map(group_sizes)
    detail["percent_of_group"] = 100 * detail["patients"] / detail["group_patients"]
    detail["small_cell"] = detail["patients"].lt(MINIMUM_GROUP_SIZE)

    summary = detail.drop_duplicates("group").drop(columns="small_cell").rename(columns={
        "source_code": "most_common_source_code",
        "patients": "patients_with_most_common_code",
        "percent_of_group": "most_common_code_percent_of_group",
    })
    summary["distinct_source_codes"] = summary["group"].map(detail["group"].value_counts())
    return summary[[
        "group", "group_patients", "distinct_source_codes", "most_common_source_code",
        "patients_with_most_common_code", "most_common_code_percent_of_group",
    ]].reset_index(drop=True), detail


def profile_diagnosis_detail(
    cohort: pd.DataFrame, source_cohort: pd.DataFrame, diagnosis_summary: pd.DataFrame,
) -> DiagnosisDetail:
    """Compare aligned diagnoses and reconcile the prepared grouping with its source codes."""
    aligned = _align_diagnoses(cohort, source_cohort)
    positions = _diagnosis_positions(aligned)
    summary, detail = _source_code_counts(positions)
    return DiagnosisDetail(
        diagnosis_position_depth=_position_depth(aligned.loc[:, list(DIAGNOSIS_GROUP_COLUMNS)]),
        diagnosis_group_gain=_group_gain(diagnosis_summary),
        diagnosis_group_pairs=_group_pairs(positions, len(aligned)),
        diagnosis_source_code_summary=summary,
        diagnosis_source_code_detail=detail,
    )
