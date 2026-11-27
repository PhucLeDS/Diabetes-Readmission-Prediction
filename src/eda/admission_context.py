"""Describe the composition and measurements of admission groups."""

from dataclasses import dataclass

import pandas as pd

from src.eda.cohort_selection import MINIMUM_GROUP_SIZE
from src.eda.data import ADMISSION_COLUMNS
from src.eda.descriptive import QUANTILES
from src.eda.nominal import AGE_BANDS
from src.preparation.data_loading import COUNT_COLUMNS


@dataclass
class AdmissionContext:
    """Admission combinations, age distributions, and count summaries."""

    admission_type_source: pd.DataFrame
    age_by_admission: pd.DataFrame
    numeric_by_admission: pd.DataFrame


def _cross_counts(
    rows: pd.Series, columns: pd.Series, row_order: pd.Index, column_order: pd.Index,
) -> pd.DataFrame:
    counts = pd.crosstab(rows, columns).reindex(index=row_order, columns=column_order, fill_value=0)
    return counts.rename_axis(index=rows.name, columns=columns.name).stack().reset_index(name="patients")


def _type_source_table(codes: pd.DataFrame, lookups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    type_column, source_column = ADMISSION_COLUMNS
    types, sources = lookups[type_column], lookups[source_column]
    table = _cross_counts(
        codes[type_column].rename("admission_type_code"), codes[source_column].rename("admission_source_code"),
        types.index, sources.index,
    )
    table = table.join(types.add_prefix("admission_type_"), on="admission_type_code", validate="many_to_one")
    table = table.join(sources.add_prefix("admission_source_"), on="admission_source_code", validate="many_to_one")
    type_patients = table.groupby("admission_type_code")["patients"].transform("sum")
    return table.assign(
        type_patients=type_patients,
        percent_of_admission_type=100 * table["patients"] / type_patients,
        small_cell=table["patients"].lt(MINIMUM_GROUP_SIZE),
    )[
        ["admission_type_code", "admission_type_description", "admission_type_lookup_status",
         "admission_source_code", "admission_source_description", "admission_source_lookup_status",
         "patients", "type_patients", "percent_of_admission_type", "small_cell"]
    ]


def _age_table(code: pd.Series, age: pd.Series, lookup: pd.DataFrame) -> pd.DataFrame:
    levels = set(age)
    age_order = pd.Index([
        *[band for band in AGE_BANDS if band in levels],
        *sorted(levels.difference(AGE_BANDS, {""})),
        *([""] if "" in levels else []),
    ])
    counts = _cross_counts(code, age, lookup.index, age_order)
    group_patients = counts.groupby("code")["patients"].transform("sum")
    return counts.join(lookup, on="code", validate="many_to_one").assign(
        group_patients=group_patients,
        percent_of_group=100 * counts["patients"] / group_patients,
        small_cell=counts["patients"].lt(MINIMUM_GROUP_SIZE),
    )


def _numeric_table(cohort: pd.DataFrame, code: pd.Series, lookup: pd.DataFrame) -> pd.DataFrame:
    values = cohort.loc[:, list(COUNT_COLUMNS)].assign(code=code).melt(
        id_vars="code", var_name="measure", value_name="value",
    )
    values["is_zero"] = values["value"].eq(0).fillna(False)
    groups = values.groupby(["code", "measure"], sort=False)
    counts = groups.agg(group_patients=("value", "size"), valid=("value", "count"), zero_count=("is_zero", "sum"))
    quartiles = {name: QUANTILES[name] for name in ("q1", "median", "q3")}
    percentiles = (
        groups["value"].quantile(list(quartiles.values()), interpolation="linear").unstack()
        .rename(columns={probability: name for name, probability in quartiles.items()})
    )
    order = pd.MultiIndex.from_product([lookup.index, COUNT_COLUMNS], names=["code", "measure"])
    table = counts.join(percentiles).reindex(order).reset_index()
    return table.join(lookup, on="code", validate="many_to_one").assign(
        missing=table["group_patients"] - table["valid"],
        zero_percent_of_valid=100 * table["zero_count"] / table["valid"].where(table["valid"].gt(0)),
        small_group=table["valid"].lt(MINIMUM_GROUP_SIZE),
    )


def profile_admission_context(cohort: pd.DataFrame, admission_descriptions: pd.DataFrame) -> AdmissionContext:
    """Profile observed codes with lookup statuses from the nominal summary."""
    lookups = {
        column: rows.set_index("code")[["description", "lookup_status"]]
        for column, rows in admission_descriptions.groupby("column", sort=False)
    }
    codes = cohort.loc[:, list(ADMISSION_COLUMNS)].fillna("")
    age = cohort["age"].fillna("")
    age_tables, numeric_tables = [], []
    for column in ADMISSION_COLUMNS:
        lookup = lookups[column]
        code = codes[column].rename("code")
        age_tables.append(_age_table(code, age, lookup).assign(column=column))
        numeric_tables.append(_numeric_table(cohort, code, lookup).assign(column=column))
    return AdmissionContext(
        admission_type_source=_type_source_table(codes, lookups),
        age_by_admission=pd.concat(age_tables, ignore_index=True)[
            ["column", "code", "description", "lookup_status", "age",
             "patients", "group_patients", "percent_of_group", "small_cell"]
        ],
        numeric_by_admission=pd.concat(numeric_tables, ignore_index=True)[
            ["column", "code", "description", "lookup_status", "measure", "group_patients",
             "valid", "missing", "q1", "median", "q3", "zero_count", "zero_percent_of_valid", "small_group"]
        ],
    )
