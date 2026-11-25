"""Analyze prior service use and a fixed history-based alert."""

from dataclasses import dataclass
from itertools import product

import pandas as pd

from src.analysis.data import AnalysisInputs
from src.analysis.outcomes import summarize_group_outcomes


UNKNOWN_HISTORY = "unknown"


@dataclass
class PriorVisitProfile:
    """The four aggregate tables used to answer Q1."""

    prior_visit_summary: pd.DataFrame
    prior_visit_combinations: pd.DataFrame
    prior_inpatient_levels: pd.DataFrame
    history_rule_confusion_matrix: pd.DataFrame


def prior_visit_groups(analysis: AnalysisInputs) -> pd.Series:
    """Classify the three service counts using the configured joint missing policy."""
    question = analysis.config["questions"]["Q1"]
    no_visits, any_visit = question["comparison_groups"]
    visits = analysis.predictors[question["visit_columns"]]
    complete = visits.notna().all(axis=1)
    has_visit = visits.gt(0).any(axis=1)
    return (
        has_visit.map({False: no_visits, True: any_visit})
        .where(complete, UNKNOWN_HISTORY)
        .astype("string")
        .rename("history_group")
    )


def _visit_combinations(visits: pd.DataFrame, history: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    flags = visits.gt(0).astype("string").replace({"False": "no", "True": "yes"})
    flags.columns = [f"prior_{column.removeprefix('number_')}" for column in visits]
    unknown = history.eq(UNKNOWN_HISTORY)
    keys = list(product(("no", "yes"), repeat=len(flags.columns)))
    if unknown.any():
        flags.loc[unknown, :] = UNKNOWN_HISTORY
        keys.append((UNKNOWN_HISTORY,) * len(flags.columns))
    return flags, pd.DataFrame(keys, columns=flags.columns)


def _inpatient_levels(counts: pd.Series, intervals: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    levels = pd.Series(UNKNOWN_HISTORY, index=counts.index, dtype="string", name="inpatient_level")
    labels = []
    for interval in intervals:
        lower, upper = interval["minimum"], interval["maximum"]
        in_interval = counts.ge(lower) if upper is None else counts.between(lower, upper)
        levels.loc[in_interval.fillna(False)] = interval["label"]
        labels.append(interval["label"])
    if (counts.notna() & levels.eq(UNKNOWN_HISTORY)).any():
        raise ValueError("The configured inpatient levels do not cover the recorded counts.")
    if counts.isna().any():
        labels.append(UNKNOWN_HISTORY)
    return levels.to_frame(), pd.DataFrame({levels.name: labels})


def _history_rule_metrics(summary: pd.DataFrame, comparison_groups: list[str]) -> pd.DataFrame:
    groups = summary.set_index("history_group")
    no_visits, any_visit = comparison_groups
    true_positives = int(groups.at[any_visit, "readmissions"])
    false_positives = int(groups.at[any_visit, "no_recorded_readmission"])
    false_negatives = int(groups.at[no_visits, "readmissions"])
    true_negatives = int(groups.at[no_visits, "no_recorded_readmission"])
    alerts = true_positives + false_positives
    evaluated_readmissions = true_positives + false_negatives
    unknown_history = groups[["patients", "readmissions"]].reindex([UNKNOWN_HISTORY], fill_value=0).iloc[0]
    return pd.DataFrame([{
        "cohort_size": int(summary["patients"].sum()),
        "evaluated_patients": alerts + false_negatives + true_negatives,
        "unknown_history_patients": int(unknown_history["patients"]),
        "unknown_history_readmissions": int(unknown_history["readmissions"]),
        "alerts": alerts,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "precision_percent": 100 * true_positives / alerts if alerts else None,
        "recall_percent": 100 * true_positives / evaluated_readmissions if evaluated_readmissions else None,
    }])


def profile_prior_visits(analysis: AnalysisInputs) -> PriorVisitProfile:
    """Compare service history groups and measure the fixed alert on complete histories."""
    question = analysis.config["questions"]["Q1"]
    thresholds = analysis.config["small_groups"]
    history = prior_visit_groups(analysis)
    comparison_groups = question["comparison_groups"]
    history_order = list(comparison_groups)
    if history.eq(UNKNOWN_HISTORY).any():
        history_order.append(UNKNOWN_HISTORY)
    summary = summarize_group_outcomes(
        history.to_frame(), pd.DataFrame({history.name: history_order}), analysis.target, **thresholds,
    )

    flags, flag_order = _visit_combinations(analysis.predictors[question["visit_columns"]], history)
    combinations = summarize_group_outcomes(flags, flag_order, analysis.target, **thresholds)
    levels, level_order = _inpatient_levels(
        analysis.predictors[question["inpatient_column"]], question["inpatient_groups"],
    )
    inpatient = summarize_group_outcomes(levels, level_order, analysis.target, **thresholds)
    return PriorVisitProfile(summary, combinations, inpatient, _history_rule_metrics(summary, comparison_groups))
