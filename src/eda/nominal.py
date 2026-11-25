"""Summarize nominal predictors in the full prepared cohort."""

from dataclasses import dataclass

import pandas as pd

from src.eda.data import ADMISSION_COLUMNS, MEDICATION_COLUMNS, EdaInputs
from src.eda.descriptive import LAB_COLUMNS, _with_percent
from src.preparation.dataset_preparation import DIAGNOSIS_GROUP_COLUMNS, DIAGNOSIS_GROUPS


DEMOGRAPHIC_COLUMNS = ("age", "race", "gender")
MEDICATION_FLAG_COLUMNS = ("change", "diabetesMed")
MEDICATION_LEVELS = ("No", "Steady", "Up", "Down")
AGE_BANDS = tuple(f"[{age}-{age + 10})" for age in range(0, 100, 10))
LAB_LEVELS = {
    "A1Cresult": ("Norm", ">7", ">8", "None"),
    "max_glu_serum": ("Norm", ">200", ">300", "None"),
}
RARE_LEVEL_PERCENT = 1.0


@dataclass
class NominalProfile:
    """Cohort frequencies and tables for interpreting each group of nominal predictors."""

    nominal_profile: pd.DataFrame
    nominal_frequency: pd.DataFrame
    admission_descriptions: pd.DataFrame
    laboratory_results: pd.DataFrame
    medication_status: pd.DataFrame
    diagnosis_summary: pd.DataFrame
    diagnosis_coverage: pd.DataFrame


def _level_order(column: str, eda: EdaInputs) -> tuple[str, ...]:
    if column == "age":
        return AGE_BANDS
    if column in ADMISSION_COLUMNS:
        return tuple(eda.admission_descriptions.loc[eda.admission_descriptions["column"].eq(column), "code"])
    if column in LAB_LEVELS:
        return LAB_LEVELS[column]
    if column in MEDICATION_COLUMNS:
        return MEDICATION_LEVELS
    if column in DIAGNOSIS_GROUP_COLUMNS:
        return DIAGNOSIS_GROUPS
    return ()


def _nominal_counts(eda: EdaInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    schema = eda.column_dictionary
    columns = schema.loc[schema["role"].eq("predictor") & schema["logical_type"].eq("nominal"), "column"]
    cohort_size = len(eda.cohort)
    tables, summaries = [], []
    for column in columns:
        counts = eda.cohort[column].fillna("").value_counts().sort_index().sort_values(ascending=False, kind="stable")
        nonempty = counts.drop(index="", errors="ignore")
        prefix = [level for level in _level_order(column, eda) if level in nonempty.index]
        tail = [level for level in nonempty.index if level not in prefix]
        levels = [*prefix, *tail, *([""] if "" in counts.index else [])]
        table = counts.reindex(levels).rename_axis("value").reset_index(name="patients").assign(column=column)
        table = _with_percent(table, cohort_size)
        table["is_empty_field"] = table["value"].eq("")
        table["is_rare_level"] = ~table["is_empty_field"] & table["percent"].lt(RARE_LEVEL_PERCENT)
        tables.append(table)
        summaries.append({
            "column": column,
            "observed_nonempty_levels": len(nonempty),
            "dominant_level": nonempty.index[0] if not nonempty.empty else pd.NA,
            "dominant_patients": int(nonempty.iloc[0]) if not nonempty.empty else 0,
            "dominant_percent_of_cohort": 100 * nonempty.iloc[0] / cohort_size if not nonempty.empty else 0.0,
            "rare_nonempty_levels": int(table["is_rare_level"].sum()),
        })
    frequency = pd.concat(tables, ignore_index=True)[
        ["column", "value", "patients", "cohort_size", "percent", "is_empty_field", "is_rare_level"]
    ]
    return pd.DataFrame(summaries), frequency


def _admission_summary(frequency: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    counts = frequency.loc[frequency["column"].isin(ADMISSION_COLUMNS)].rename(columns={"value": "code"})
    table = counts.merge(
        lookup[["column", "code", "description", "lookup_status"]],
        on=["column", "code"], how="left", validate="one_to_one",
    )
    return table.assign(
        description=table["description"].fillna(""),
        lookup_status=table["lookup_status"].fillna("Unmapped code").mask(table["is_empty_field"], "Empty CSV field"),
    )[["column", "code", "description", "lookup_status", "patients", "cohort_size", "percent", "is_rare_level"]]


def _laboratory_summary(frequency: pd.DataFrame) -> pd.DataFrame:
    table = frequency.loc[frequency["column"].isin(LAB_COLUMNS)].copy()
    recorded = ~table["is_empty_field"] & table["value"].ne("None")
    totals = table["patients"].where(recorded, 0).groupby(table["column"]).transform("sum")
    table["result_recorded"] = recorded
    table["patients_with_result"] = totals
    table["recorded_percent_of_cohort"] = 100 * totals / table["cohort_size"]
    table["percent_of_recorded"] = 100 * table["patients"].where(recorded) / totals.where(totals.gt(0))
    return table.drop(columns=["is_empty_field", "is_rare_level"])


def _medication_summary(frequency: pd.DataFrame, cohort_size: int) -> pd.DataFrame:
    measures = frequency.loc[frequency["column"].isin(MEDICATION_COLUMNS)]
    counts = measures.pivot(index="column", columns="value", values="patients")
    columns = [*MEDICATION_LEVELS, *(level for level in counts.columns if level not in MEDICATION_LEVELS)]
    counts = counts.reindex(index=MEDICATION_COLUMNS, columns=columns).fillna(0).astype("int64")

    # Retain any unexpected level in the frequency table and reconcile it in this matrix.
    other_columns = [level for level in columns if level not in (*MEDICATION_LEVELS, "")]
    patients = counts[list(MEDICATION_LEVELS)].rename(columns=lambda level: f"{level.lower()}_patients")
    patients = patients.assign(
        any_non_no_patients=measures.loc[~measures["is_empty_field"] & measures["value"].ne("No")]
        .groupby("column")["patients"].sum().reindex(MEDICATION_COLUMNS, fill_value=0),
        empty_patients=counts[""] if "" in counts else 0,
        other_status_patients=counts[other_columns].sum(axis=1) if other_columns else 0,
    )
    percentages = (100 * patients / cohort_size).rename(columns=lambda name: name.replace("_patients", "_percent"))
    return pd.concat([patients, percentages], axis=1).assign(cohort_size=cohort_size).reset_index()


def _diagnosis_summaries(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnoses = cohort.loc[:, list(DIAGNOSIS_GROUP_COLUMNS)].fillna("")
    observed = set(diagnoses.to_numpy().ravel())
    groups = [*DIAGNOSIS_GROUPS, *sorted(observed.difference(DIAGNOSIS_GROUPS))]
    rows = []
    for group in groups:
        matches = diagnoses.eq(group)
        rows.append({
            "group": group,
            **{f"{column}_patients": int(matches[column].sum()) for column in DIAGNOSIS_GROUP_COLUMNS},
            "patients_at_any_position": int(matches.any(axis=1).sum()),
        })
    summary = pd.DataFrame(rows).assign(cohort_size=len(cohort))
    summary["percent_at_any_position"] = 100 * summary["patients_at_any_position"] / len(cohort)

    mapped = ~diagnoses.isin(["", "Missing", "Unmapped"])
    coverage = (
        mapped.sum(axis=1).value_counts().reindex(range(4), fill_value=0)
        .rename_axis("positions_with_mapped_codes").reset_index(name="patients")
    )
    return summary, _with_percent(coverage, len(cohort))


def profile_nominal(eda: EdaInputs) -> NominalProfile:
    """Summarize observed nominal levels, keeping literal source categories intact."""
    summary, frequency = _nominal_counts(eda)
    diagnoses, coverage = _diagnosis_summaries(eda.cohort)
    return NominalProfile(
        nominal_profile=summary,
        nominal_frequency=frequency,
        admission_descriptions=_admission_summary(frequency, eda.admission_descriptions),
        laboratory_results=_laboratory_summary(frequency),
        medication_status=_medication_summary(frequency, len(eda.cohort)),
        diagnosis_summary=diagnoses,
        diagnosis_coverage=coverage,
    )
