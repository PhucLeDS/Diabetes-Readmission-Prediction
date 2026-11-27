"""Describe recorded medication combinations and compare them with source flags."""

from dataclasses import dataclass

import pandas as pd

from src.eda.cohort_selection import MINIMUM_GROUP_SIZE
from src.eda.data import MEDICATION_COLUMNS
from src.eda.nominal import MEDICATION_LEVELS
from src.preparation.data_loading import ID_COLUMNS


EXCLUDED_MEDICATION_COLUMNS = (
    "acetohexamide", "tolbutamide", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
)
ADJUSTED_STATUSES = ("Up", "Down")
RECORDED_STATUSES = ("Steady", *ADJUSTED_STATUSES)


@dataclass
class MedicationDetail:
    """Aggregate results for the retained fields and the excluded source fields."""

    medication_count_distribution: pd.DataFrame
    insulin_medication_pairs: pd.DataFrame
    medication_flag_reconciliation: pd.DataFrame


def retained_medication_flag_measures(statuses: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the count of nonempty non-No statuses and the Up or Down indicator."""
    non_no_count = (~statuses.fillna("").isin(["", "No"])).sum(axis=1)
    has_up_or_down = statuses.isin(ADJUSTED_STATUSES).any(axis=1)
    return non_no_count, has_up_or_down


def _medication_distribution(recorded: pd.DataFrame, adjusted: pd.DataFrame, unknown: pd.DataFrame) -> pd.DataFrame:
    count_levels = pd.RangeIndex(len(MEDICATION_COLUMNS) + 1, name="medications")
    coverage_unknown = unknown.any(axis=1)
    tables = []
    for measure, matches in (("recorded_status", recorded), ("up_or_down", adjusted)):
        per_patient = matches.sum(axis=1)
        counts = per_patient.value_counts().reindex(count_levels, fill_value=0)
        uncertain = per_patient.loc[coverage_unknown].value_counts().reindex(count_levels, fill_value=0)
        tables.append(pd.DataFrame({
            "measure": measure,
            "patients": counts,
            "cohort_size": len(matches),
            "percent_of_cohort": 100 * counts / len(matches),
            "patients_with_unknown_retained_status": uncertain,
            "small_cell": counts.lt(MINIMUM_GROUP_SIZE),
        }).reset_index())
    return pd.concat(tables, ignore_index=True)[
        ["measure", "medications", "patients", "cohort_size", "percent_of_cohort",
         "patients_with_unknown_retained_status", "small_cell"]
    ]


def _insulin_pairs(recorded: pd.DataFrame, unknown: pd.DataFrame) -> pd.DataFrame:
    insulin_mask = recorded["insulin"]
    drugs = [drug for drug in MEDICATION_COLUMNS if drug != "insulin"]
    insulin_patients = int(insulin_mask.sum())
    pair_counts = recorded.loc[insulin_mask, drugs].sum()
    return pd.DataFrame({
        "medication": drugs,
        "patients_with_medication": recorded[drugs].sum().to_numpy(),
        "pair_patients": pair_counts.to_numpy(),
        "insulin_patients": insulin_patients,
        "cohort_size": len(recorded),
        "percent_of_insulin_patients": (100 * pair_counts / (insulin_patients or float("nan"))).to_numpy(),
        "percent_of_cohort": (100 * pair_counts / len(recorded)).to_numpy(),
        "insulin_patients_with_unknown_medication_status": unknown.loc[insulin_mask, drugs].sum().to_numpy(),
        "small_cell": pair_counts.lt(MINIMUM_GROUP_SIZE).to_numpy(),
    }).sort_values("pair_patients", ascending=False, kind="stable", ignore_index=True)


def _flag_reconciliation(
    cohort: pd.DataFrame, retained: pd.DataFrame, excluded: pd.DataFrame, unknown: pd.DataFrame,
) -> pd.DataFrame:
    non_no_count, has_up_or_down = retained_medication_flag_measures(retained)
    comparisons = (
        ("diabetesMed", "Yes", "no recorded non-No status", non_no_count.eq(0), RECORDED_STATUSES),
        ("change", "Ch", "no recorded Up or Down status", ~has_up_or_down, ADJUSTED_STATUSES),
    )
    records = []
    for flag, level, criterion, no_retained_evidence, matching_statuses in comparisons:
        flag_group = cohort[flag].eq(level).fillna(False)
        compare = flag_group & no_retained_evidence
        excluded_cases = excluded.loc[compare]
        matches = excluded_cases.isin(matching_statuses)
        unknown_excluded = ~excluded_cases.isin(MEDICATION_LEVELS).all(axis=1)
        states = pd.Series("no_matching_status_recorded", index=excluded_cases.index, dtype="string")
        states.loc[unknown_excluded] = "unknown"
        states.loc[matches.any(axis=1)] = "matching_status_recorded"
        unknown_retained = unknown.loc[compare].any(axis=1)
        for state in ("matching_status_recorded", "no_matching_status_recorded", "unknown"):
            mask = states.eq(state)
            patients = int(mask.sum())
            matching_columns = matches.columns[matches.loc[mask].any(axis=0)]
            records.append({
                "source_flag": flag,
                "flag_value": level,
                "retained_criterion": criterion,
                "flag_group_patients": int(flag_group.sum()),
                "comparison_patients": len(states),
                "excluded_status": state,
                "patients": patients,
                "percent_of_comparison_group": 100 * patients / (len(states) or float("nan")),
                "matching_excluded_columns": ", ".join(matching_columns),
                "patients_with_unknown_retained_status": int((mask & unknown_retained).sum()),
                "patients_with_unknown_excluded_status": int((mask & unknown_excluded).sum()),
                "small_cell": patients < MINIMUM_GROUP_SIZE,
            })
    return pd.DataFrame.from_records(records)


def profile_medication_detail(cohort: pd.DataFrame, source_cohort: pd.DataFrame) -> MedicationDetail:
    """Summarize retained medication statuses and check matching excluded columns."""
    keys = pd.MultiIndex.from_frame(cohort.loc[:, list(ID_COLUMNS)])
    excluded = source_cohort.loc[:, [*ID_COLUMNS, *EXCLUDED_MEDICATION_COLUMNS]].set_index(
        list(ID_COLUMNS),
    )
    if (
        not keys.is_unique
        or not excluded.index.is_unique
        or len(keys) != len(excluded)
        or not keys.isin(excluded.index).all()
    ):
        raise ValueError("Prepared and source medication encounter-patient keys differ.")
    excluded = excluded.loc[keys].set_axis(cohort.index)
    retained = cohort.loc[:, list(MEDICATION_COLUMNS)]
    recorded = retained.isin(RECORDED_STATUSES)
    adjusted = retained.isin(ADJUSTED_STATUSES)
    unknown = ~retained.isin(MEDICATION_LEVELS)
    return MedicationDetail(
        medication_count_distribution=_medication_distribution(recorded, adjusted, unknown),
        insulin_medication_pairs=_insulin_pairs(recorded, unknown),
        medication_flag_reconciliation=_flag_reconciliation(cohort, retained, excluded, unknown),
    )
