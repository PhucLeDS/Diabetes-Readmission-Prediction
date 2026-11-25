"""Compare current encounter characteristics within prior service groups."""

from dataclasses import dataclass
from itertools import product

import pandas as pd

from src.analysis.data import AnalysisInputs
from src.analysis.outcomes import summarize_group_outcomes
from src.analysis.prior_visits import UNKNOWN_HISTORY, prior_visit_groups


MISSING_STAY = "missing"
OUTSIDE_STAY_RANGE = "outside_configured_range"


@dataclass
class CurrentEncounterProfile:
    """The outcome tables for length of stay and diagnosis information."""

    stay_by_history: pd.DataFrame
    diagnoses_count_by_history: pd.DataFrame
    primary_diagnosis_by_history: pd.DataFrame


def stay_groups(days: pd.Series, intervals: list[dict]) -> pd.Series:
    """Assign inclusive stay intervals and identify missing or out-of-range values."""
    ordered = sorted(intervals, key=lambda interval: interval["minimum"])
    if (
        any(interval["minimum"] > interval["maximum"] for interval in ordered)
        or any(left["maximum"] >= right["minimum"] for left, right in zip(ordered, ordered[1:]))
    ):
        raise ValueError("Configured stay intervals must have valid bounds and must not overlap.")

    groups = pd.Series(OUTSIDE_STAY_RANGE, index=days.index, dtype="string", name="stay_group")
    groups.loc[days.isna()] = MISSING_STAY
    for interval in intervals:
        in_interval = days.between(interval["minimum"], interval["maximum"]).fillna(False)
        groups.loc[in_interval] = interval["label"]
    return groups


def profile_current_encounters(analysis: AnalysisInputs) -> CurrentEncounterProfile:
    """Summarize all three characteristics using the same Q1 history assignments."""
    question = analysis.config["questions"]["Q2"]
    history = prior_visit_groups(analysis)
    history_order = [question["focus_group"], question["comparison_group"]]
    if history.eq(UNKNOWN_HISTORY).any():
        history_order.append(UNKNOWN_HISTORY)

    stays = stay_groups(analysis.predictors[question["stay_column"]], question["stay_groups"])
    stay_order = [interval["label"] for interval in question["stay_groups"]]
    stay_order.extend(label for label in (MISSING_STAY, OUTSIDE_STAY_RANGE) if stays.eq(label).any())

    diagnosis_counts = analysis.predictors[question["diagnoses_count_column"]]
    count_order = diagnosis_counts.drop_duplicates().sort_values().tolist()
    primary_diagnosis = (
        analysis.predictors[question["primary_diagnosis_column"]].fillna("").replace("", "Blank field")
    )
    diagnosis_order = sorted(primary_diagnosis.unique())

    def summarize(values: pd.Series, value_order: list) -> pd.DataFrame:
        columns = [history.name, values.name]
        groups = pd.concat([history, values], axis=1)
        order = pd.DataFrame(product(history_order, value_order), columns=columns)
        order[values.name] = order[values.name].astype(values.dtype)
        return summarize_group_outcomes(groups, order, analysis.target, **analysis.config["small_groups"])

    return CurrentEncounterProfile(
        stay_by_history=summarize(stays, stay_order),
        diagnoses_count_by_history=summarize(diagnosis_counts, count_order),
        primary_diagnosis_by_history=summarize(primary_diagnosis, diagnosis_order),
    )
