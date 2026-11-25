"""Describe completeness and numeric distributions in the prepared cohort."""

from dataclasses import dataclass

import pandas as pd

from src.eda.data import ADMISSION_COLUMNS
from src.preparation.data_loading import COUNT_COLUMNS
from src.preparation.dataset_preparation import DIAGNOSIS_GROUP_COLUMNS


LAB_COLUMNS = ("A1Cresult", "max_glu_serum")
PRIOR_VISIT_COLUMNS = ("number_outpatient", "number_emergency", "number_inpatient")
QUANTILES = {"q1": 0.25, "median": 0.5, "q3": 0.75, "p95": 0.95, "p99": 0.99}


@dataclass
class CompletenessProfile:
    """Counts of empty fields and source categories that need separate interpretation."""

    missing_summary: pd.DataFrame
    missing_patterns: pd.DataFrame
    special_values: pd.DataFrame
    lab_recording_patterns: pd.DataFrame
    empty_field_count: int
    patients_with_empty_fields: int


@dataclass
class NumericProfile:
    """Summaries and exact frequencies for the numeric predictors."""

    numeric_profile: pd.DataFrame
    numeric_frequency: pd.DataFrame
    prior_visit_positive_profile: pd.DataFrame
    prior_visit_coverage: pd.DataFrame


def _with_percent(table: pd.DataFrame, cohort_size: int) -> pd.DataFrame:
    return table.assign(cohort_size=cohort_size, percent=100 * table["patients"] / cohort_size)


def _empty_fields(cohort: pd.DataFrame) -> pd.DataFrame:
    return cohort.isna() | cohort.eq("").fillna(False)


def _missing_patterns(empty: pd.DataFrame) -> pd.DataFrame:
    affected = empty.loc[:, empty.any(axis=0)]
    groups = []
    if affected.shape[1]:
        for pattern, patients in affected.value_counts(sort=False).items():
            columns = [column for column, is_empty in zip(affected.columns, pattern, strict=True) if is_empty]
            groups.append((" + ".join(columns) or "No empty fields", len(columns), patients))
    else:
        groups.append(("No empty fields", 0, len(empty)))
    table = pd.DataFrame(groups, columns=["pattern", "empty_fields_per_patient", "patients"])
    return _with_percent(table, len(empty)).sort_values(
        ["empty_fields_per_patient", "pattern"], ignore_index=True,
    )


def profile_completeness(cohort: pd.DataFrame, admission_descriptions: pd.DataFrame) -> CompletenessProfile:
    """Summarize empty fields separately from laboratory, diagnosis, and lookup categories."""
    cohort_size = len(cohort)
    empty = _empty_fields(cohort)
    empty_counts = empty.sum()
    missing = pd.DataFrame({
        "column": cohort.columns,
        "empty_fields": empty_counts.to_numpy(),
        "nonempty_fields": (cohort_size - empty_counts).to_numpy(),
        "cohort_size": cohort_size,
        "empty_percent": (100 * empty_counts / cohort_size).to_numpy(),
    })

    recorded = ~empty.loc[:, list(LAB_COLUMNS)] & cohort.loc[:, list(LAB_COLUMNS)].ne("None").fillna(False)
    lab_patterns = (
        recorded.value_counts()
        .reindex(
            pd.MultiIndex.from_product([[True, False], [True, False]], names=LAB_COLUMNS),
            fill_value=0,
        )
        .rename_axis(index=["a1c_result_recorded", "glucose_result_recorded"])
        .reset_index(name="patients")
    )

    special = []
    for column in LAB_COLUMNS:
        special.append((column, "None", cohort[column].eq("None").sum()))
        special.append((column, "Recorded result", recorded[column].sum()))
    for column in DIAGNOSIS_GROUP_COLUMNS:
        for category in ("Missing", "Unmapped"):
            special.append((column, category, cohort[column].eq(category).sum()))
    for column in ADMISSION_COLUMNS:
        lookup = admission_descriptions.loc[admission_descriptions["column"].eq(column)]
        for status in ("Mapped", "Unclear description", "Unmapped code"):
            codes = lookup.loc[lookup["lookup_status"].eq(status), "code"]
            special.append((column, status, cohort[column].isin(codes).sum()))
    special_values = pd.DataFrame(special, columns=["column", "category", "patients"])

    return CompletenessProfile(
        missing_summary=missing,
        missing_patterns=_missing_patterns(empty),
        special_values=_with_percent(special_values, cohort_size),
        lab_recording_patterns=_with_percent(lab_patterns, cohort_size),
        empty_field_count=int(empty_counts.sum()),
        patients_with_empty_fields=int(empty.any(axis=1).sum()),
    )


def _numeric_summary(values: pd.Series) -> dict:
    valid = values.dropna()
    zeros = int(valid.eq(0).sum())
    quantiles = valid.quantile(list(QUANTILES.values()), interpolation="linear")
    return {
        "valid": len(valid),
        "missing": len(values) - len(valid),
        "zero_count": zeros,
        "zero_percent_of_valid": 100 * zeros / len(valid) if len(valid) else pd.NA,
        "min": valid.min(),
        **{name: quantiles.loc[q] for name, q in QUANTILES.items()},
        "max": valid.max(),
    }


def profile_numeric(cohort: pd.DataFrame) -> NumericProfile:
    """Return numeric profiles, exact counts, and the extent of prior service use."""
    summary = []
    frequencies = []
    for column in COUNT_COLUMNS:
        values = cohort[column]
        summary.append({"column": column, **_numeric_summary(values)})
        frequency = values.value_counts(sort=False).sort_index().rename_axis("value").reset_index(name="patients")
        frequencies.append(frequency.assign(column=column, valid=int(values.count())))

    positive = []
    for column in PRIOR_VISIT_COLUMNS:
        values = cohort[column]
        stats = _numeric_summary(values.loc[values.gt(0)])
        positive.append({
            "column": column,
            "positive_count": stats["valid"],
            "min": stats["min"],
            **{name: stats[name] for name in QUANTILES},
            "max": stats["max"],
        })

    prior = cohort.loc[:, list(PRIOR_VISIT_COLUMNS)]
    # A missing count makes the exact number of prior service types unknown.
    service_types = prior.gt(0).sum(axis=1, skipna=False)
    coverage = (
        service_types.value_counts(dropna=False)
        .reindex(pd.Index([0, 1, 2, 3, pd.NA], dtype="Int64"), fill_value=0)
        .rename_axis("service_types_with_prior_visits")
        .reset_index(name="patients")
    )
    return NumericProfile(
        numeric_profile=pd.DataFrame(summary),
        numeric_frequency=pd.concat(frequencies, ignore_index=True)[["column", "value", "patients", "valid"]],
        prior_visit_positive_profile=pd.DataFrame(positive),
        prior_visit_coverage=_with_percent(coverage, len(cohort)),
    )
