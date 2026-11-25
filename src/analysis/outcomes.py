"""Summarize recorded readmissions for a partition of the analysis cohort."""

import pandas as pd

from src.preparation.dataset_export import CLASS_LABELS


def summarize_group_outcomes(
    groups: pd.DataFrame,
    group_order: pd.DataFrame,
    target: pd.Series,
    *,
    minimum_patients: int,
    minimum_per_class: int,
) -> pd.DataFrame:
    """Count both outcomes per group and calculate rates with explicit denominators."""
    if not groups.index.equals(target.index):
        raise ValueError("Group assignments and outcomes must refer to the same rows.")

    group_columns = list(groups.columns)
    positive = target.eq(CLASS_LABELS[1])
    observed = (
        groups.assign(readmissions=positive)
        .groupby(group_columns, sort=False, dropna=False)["readmissions"]
        .agg(patients="size", readmissions="sum")
        .reset_index()
    )
    summary = group_order.merge(observed, on=group_columns, how="left", validate="one_to_one")
    for column in ("patients", "readmissions"):
        summary[column] = summary[column].fillna(0).astype("int64")
    if summary["patients"].sum() != len(target):
        raise ValueError("The configured groups do not cover every patient in the analysis.")

    summary["no_recorded_readmission"] = summary["patients"] - summary["readmissions"]
    summary["cohort_size"] = len(target)
    summary["cohort_readmissions"] = int(positive.sum())
    summary["coverage_percent"] = 100 * summary["patients"] / summary["cohort_size"].replace(0, float("nan"))
    summary["readmission_rate_percent"] = 100 * summary["readmissions"] / summary["patients"].replace(0, float("nan"))
    summary["share_of_readmissions_percent"] = (
        100 * summary["readmissions"] / summary["cohort_readmissions"].replace(0, float("nan"))
    )
    summary["small_group"] = (
        summary["patients"].lt(minimum_patients)
        | summary["readmissions"].lt(minimum_per_class)
        | summary["no_recorded_readmission"].lt(minimum_per_class)
    )
    return summary
