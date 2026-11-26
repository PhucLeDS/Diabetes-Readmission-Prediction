"""Describe selected relationships and the two representations of the readmission label."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.eda.data import MEDICATION_COLUMNS
from src.eda.descriptive import PRIOR_VISIT_COLUMNS, QUANTILES, _with_percent
from src.eda.nominal import AGE_BANDS
from src.preparation.data_loading import TARGET_COLUMN, TARGET_LABELS
from src.preparation.dataset_export import (
    CLASS_LABELS, TARGET_MAPPING_CONFIG_PATH, load_target_mapping, map_readmission_labels,
)


STAY_COLUMN = "time_in_hospital"
STAY_SERVICE_COLUMNS = ("num_lab_procedures", "num_procedures", "num_medications")


@dataclass
class RelationshipProfile:
    """Descriptive tables for the full prepared cohort."""

    stay_service_summary: pd.DataFrame
    age_prior_visits: pd.DataFrame
    medication_flags: pd.DataFrame
    outcome_distribution: pd.DataFrame


def _stay_services(cohort: pd.DataFrame) -> pd.DataFrame:
    values = cohort.melt(
        id_vars=STAY_COLUMN, value_vars=STAY_SERVICE_COLUMNS, var_name="measure", value_name="value",
    )
    groups = values.groupby([STAY_COLUMN, "measure"], sort=False, dropna=False)["value"]
    quartiles = {name: QUANTILES[name] for name in ("q1", "median", "q3")}
    percentiles = (
        groups.quantile(list(quartiles.values()), interpolation="linear").unstack()
        .rename(columns={probability: name for name, probability in quartiles.items()})
    )
    return (
        groups.agg(patients="size", valid="count").join(percentiles).reset_index()
        .sort_values(STAY_COLUMN, kind="stable", ignore_index=True)
    )


def _age_prior_visits(cohort: pd.DataFrame) -> pd.DataFrame:
    values = cohort.assign(age=cohort["age"].fillna("")).melt(
        id_vars="age", value_vars=PRIOR_VISIT_COLUMNS, var_name="service", value_name="visits",
    )
    values["has_prior_visit"] = values["visits"].gt(0)
    table = (
        values.groupby(["age", "service"], sort=False)
        .agg(patients=("visits", "size"), valid=("visits", "count"), positive=("has_prior_visit", "sum"))
        .reset_index()
    )
    order = [*AGE_BANDS, *sorted(set(values["age"]).difference(AGE_BANDS, {""})), ""]
    age_order = pd.Series(range(len(order)), index=order)
    table["percent_of_age_group"] = 100 * table["positive"] / table["patients"]
    return table.sort_values("age", key=lambda values: values.map(age_order), kind="stable", ignore_index=True)


def _medication_flags(cohort: pd.DataFrame) -> pd.DataFrame:
    statuses = cohort.loc[:, list(MEDICATION_COLUMNS)].fillna("")
    non_no_count = (~statuses.isin(["", "No"])).sum(axis=1)
    has_up_or_down = statuses.isin(["Up", "Down"]).any(axis=1).astype(int)
    comparisons = (
        ("diabetesMed", "Retained medications with a recorded non-No status", non_no_count),
        ("change", "Any retained medication with Up or Down (0 = no, 1 = yes)", has_up_or_down),
    )
    tables = []
    for flag, measure, retained_status in comparisons:
        groups = pd.DataFrame({
            "flag_value": cohort[flag].fillna(""),
            "retained_status": retained_status,
        })
        counts = groups.value_counts(sort=False).sort_index().reset_index(name="patients")
        counts["flag_group_size"] = counts.groupby("flag_value")["patients"].transform("sum")
        counts["percent_of_flag_group"] = 100 * counts["patients"] / counts["flag_group_size"]
        counts = _with_percent(counts, len(cohort)).rename(columns={"percent": "percent_of_cohort"})
        tables.append(counts.assign(source_flag=flag, measure=measure))
    return pd.concat(tables, ignore_index=True)[
        ["source_flag", "measure", "flag_value", "retained_status", "patients", "flag_group_size",
         "cohort_size", "percent_of_flag_group", "percent_of_cohort"]
    ]


def _outcome_distribution(cohort: pd.DataFrame, target_config_path: Path) -> pd.DataFrame:
    mapping = load_target_mapping(target_config_path)
    original = cohort[TARGET_COLUMN]
    binary = map_readmission_labels(original, mapping)
    original_counts = original.value_counts()
    binary_counts = binary.value_counts()
    rows = [
        ("Original", label, mapping[label], original_counts.get(label, 0))
        for label in TARGET_LABELS
    ]
    rows.extend(("Any Readmission", label, label, binary_counts.get(label, 0)) for label in CLASS_LABELS)
    table = pd.DataFrame(rows, columns=["representation", "label", "any_readmission_class", "patients"])
    return _with_percent(table, len(cohort))


def profile_relationships(
    cohort: pd.DataFrame, *, target_config_path: Path = TARGET_MAPPING_CONFIG_PATH,
) -> RelationshipProfile:
    """Describe covariates together and summarize outcome labels separately."""
    return RelationshipProfile(
        stay_service_summary=_stay_services(cohort),
        age_prior_visits=_age_prior_visits(cohort),
        medication_flags=_medication_flags(cohort),
        outcome_distribution=_outcome_distribution(cohort, target_config_path),
    )
