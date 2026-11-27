"""Describe how the prepared cohort was selected from the cleaned encounters."""

from dataclasses import dataclass

import pandas as pd

from src.eda.data import EdaInputs
from src.preparation.dataset_preparation import DISCHARGE_COLUMN
from src.validation import validate_partitions


PROFILE_ATTRIBUTES = ("age", "admission_type_id", "time_in_hospital")
MINIMUM_GROUP_SIZE = 30


@dataclass
class CohortSelectionProfile:
    """Aggregate comparisons with explicit encounter or patient denominators."""

    discharge_eligibility_profile: pd.DataFrame
    eligible_encounters_per_patient: pd.DataFrame
    representative_encounter_profile: pd.DataFrame
    discharge_patient_overlap: pd.DataFrame


def _encounter_profile(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    tables = []
    for group, frame in groups.items():
        for attribute in PROFILE_ATTRIBUTES:
            counts = frame[attribute].value_counts(dropna=False).sort_index()
            table = counts.rename_axis("value").reset_index(name="encounters")
            table["value"] = table["value"].astype("string").fillna("")
            tables.append(table.assign(
                group=group,
                attribute=attribute,
                group_encounters=len(frame),
                percent_of_group=100 * table["encounters"] / len(frame),
                small_cell=table["encounters"].lt(MINIMUM_GROUP_SIZE),
            ))
    return pd.concat(tables, ignore_index=True)[
        ["group", "attribute", "value", "encounters", "group_encounters", "percent_of_group", "small_cell"]
    ]


def profile_cohort_selection(eda: EdaInputs) -> CohortSelectionProfile:
    """Compare eligibility groups and eligible encounters from patients with multiple records."""
    cleaned = eda.cleaned_encounters
    excluded_mask = cleaned[DISCHARGE_COLUMN].isin(eda.excluded_discharge_codes)
    excluded, eligible = cleaned.loc[excluded_mask], cleaned.loc[~excluded_mask]
    patient_encounters = eligible["patient_nbr"].value_counts()
    multi_visit_patients = patient_encounters.index[patient_encounters.gt(1)]
    is_representative = eligible["encounter_id"].isin(eda.source_cohort["encounter_id"])
    representatives = eligible.loc[is_representative]
    source_keys = eda.source_cohort[["encounter_id", "patient_nbr"]].itertuples(index=False, name=None)
    eligible_keys = representatives[["encounter_id", "patient_nbr"]].itertuples(index=False, name=None)
    validate_partitions(source_keys, {"eligible representatives": eligible_keys})

    stages = (cleaned, eligible, representatives)
    observed_flow = [(len(frame), frame["patient_nbr"].nunique()) for frame in stages]
    manifest_flow = list(eda.cohort_flow[["encounters", "patients"]].itertuples(index=False, name=None))
    if observed_flow != manifest_flow:
        raise ValueError("The selection counts differ from the preparation manifest.")

    distribution = (
        patient_encounters.value_counts().sort_index()
        .rename_axis("eligible_encounters_per_patient").reset_index(name="patients")
    )
    distribution = distribution.assign(
        eligible_encounters=distribution["eligible_encounters_per_patient"] * distribution["patients"],
        eligible_patients=len(patient_encounters),
        percent_of_eligible_patients=100 * distribution["patients"] / len(patient_encounters),
        small_cell=distribution["patients"].lt(MINIMUM_GROUP_SIZE),
    )

    eligible_patients = set(patient_encounters.index)
    excluded_patients = set(excluded["patient_nbr"])
    overlap = pd.DataFrame({
        "group": ["eligible_only", "eligible_and_excluded", "excluded_only"],
        "patients": [
            len(eligible_patients - excluded_patients),
            len(eligible_patients & excluded_patients),
            len(excluded_patients - eligible_patients),
        ],
        "cleaned_patients": observed_flow[0][1],
    })
    overlap["percent_of_cleaned_patients"] = 100 * overlap["patients"] / overlap["cleaned_patients"]
    overlap["small_cell"] = overlap["patients"].lt(MINIMUM_GROUP_SIZE)

    has_multiple = eligible["patient_nbr"].isin(multi_visit_patients)
    return CohortSelectionProfile(
        discharge_eligibility_profile=_encounter_profile({"eligible": eligible, "excluded": excluded}),
        eligible_encounters_per_patient=distribution,
        representative_encounter_profile=_encounter_profile({
            "representative": eligible.loc[has_multiple & is_representative],
            "other_eligible": eligible.loc[has_multiple & ~is_representative],
        }),
        discharge_patient_overlap=overlap,
    )
