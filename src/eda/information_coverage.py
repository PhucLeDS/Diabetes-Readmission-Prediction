"""Describe how recorded laboratory results overlap with other cohort information."""

from dataclasses import dataclass

import pandas as pd

from src.eda.cohort_selection import MINIMUM_GROUP_SIZE
from src.eda.data import ADMISSION_COLUMNS
from src.eda.descriptive import PRIOR_VISIT_COLUMNS
from src.eda.laboratory import laboratory_result_recorded


PRIOR_HISTORY_GROUPS = ("no_recorded_prior_visits", "any_recorded_prior_visit", "unknown")
LAB_RECORDING_PATTERNS = ("neither", "a1c_only", "glucose_only", "both", "unknown")
COVERAGE_ATTRIBUTES = ("age", *ADMISSION_COLUMNS, "time_in_hospital")


@dataclass
class InformationCoverage:
    """Laboratory recording counts with denominators for each subgroup."""

    lab_recording_by_prior_history: pd.DataFrame
    lab_recording_by_characteristic: pd.DataFrame


def _prior_history(cohort: pd.DataFrame) -> pd.Series:
    visits = cohort.loc[:, list(PRIOR_VISIT_COLUMNS)]
    history = pd.Series(PRIOR_HISTORY_GROUPS[0], index=cohort.index, name="prior_history", dtype="string")
    history.loc[visits.gt(0).any(axis=1)] = PRIOR_HISTORY_GROUPS[1]
    history.loc[visits.isna().any(axis=1)] = PRIOR_HISTORY_GROUPS[2]
    return history


def _laboratory_recording(cohort: pd.DataFrame) -> pd.Series:
    a1c_recorded = laboratory_result_recorded(cohort["A1Cresult"], "A1Cresult")
    glucose_recorded = laboratory_result_recorded(cohort["max_glu_serum"], "max_glu_serum")
    patterns = pd.Series("neither", index=cohort.index, name="lab_recording_pattern", dtype="string")
    patterns.loc[a1c_recorded & ~glucose_recorded] = "a1c_only"
    patterns.loc[~a1c_recorded & glucose_recorded] = "glucose_only"
    patterns.loc[a1c_recorded & glucose_recorded] = "both"
    patterns.loc[a1c_recorded.isna() | glucose_recorded.isna()] = "unknown"
    return patterns


def _recording_table(groups: pd.Series, patterns: pd.Series, order) -> pd.DataFrame:
    counts = pd.crosstab(groups, patterns).reindex(
        index=pd.Index(order, name=groups.name), columns=LAB_RECORDING_PATTERNS, fill_value=0,
    )
    table = counts.rename_axis(columns=patterns.name).stack().reset_index(name="patients")
    group_patients = table.groupby(groups.name)["patients"].transform("sum")
    return table.assign(
        group_patients=group_patients,
        percent_of_group=100 * table["patients"] / group_patients.where(group_patients.gt(0)),
        small_cell=table["patients"].lt(MINIMUM_GROUP_SIZE),
    )


def profile_information_coverage(cohort: pd.DataFrame) -> InformationCoverage:
    """Describe recording patterns by prior visits, age, admission codes, and exact stay length."""
    patterns = _laboratory_recording(cohort)
    history = _recording_table(_prior_history(cohort), patterns, PRIOR_HISTORY_GROUPS)
    comparisons = []
    for column in COVERAGE_ATTRIBUTES:
        values = cohort[column].astype("string").fillna("").rename("value")
        order = cohort[column].sort_values(kind="stable").astype("string").fillna("").drop_duplicates()
        comparisons.append(_recording_table(values, patterns, order).assign(characteristic=column))

    characteristics = pd.concat(comparisons, ignore_index=True)[
        ["characteristic", "value", "lab_recording_pattern", "patients",
         "group_patients", "percent_of_group", "small_cell"]
    ]
    return InformationCoverage(
        lab_recording_by_prior_history=history, lab_recording_by_characteristic=characteristics,
    )
