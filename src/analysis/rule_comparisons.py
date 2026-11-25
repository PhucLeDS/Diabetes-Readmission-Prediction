"""Compare prespecified readmission rules and their individual conditions."""

import pandas as pd

from src.analysis.current_encounters import MISSING_STAY, stay_groups
from src.analysis.data import AnalysisInputs
from src.analysis.prior_visits import UNKNOWN_HISTORY, prior_visit_groups
from src.preparation.dataset_export import CLASS_LABELS


def _rule_counts(condition: pd.Series, positive: pd.Series, thresholds: dict) -> dict:
    """Count confirmed matches and use the full cohort for coverage and support."""
    matches = condition.fillna(False)
    patients = int(matches.sum())
    readmissions = int((matches & positive).sum())
    no_readmission = patients - readmissions
    cohort_size = len(positive)
    cohort_readmissions = int(positive.sum())
    confidence = readmissions / patients if patients else None
    return {
        "matched_patients": patients,
        "matched_readmissions": readmissions,
        "matched_no_recorded_readmission": no_readmission,
        "unknown_condition_patients": int(condition.isna().sum()),
        "cohort_size": cohort_size,
        "cohort_readmissions": cohort_readmissions,
        "coverage_percent": 100 * patients / cohort_size if cohort_size else None,
        "support_percent": 100 * readmissions / cohort_size if cohort_size else None,
        "confidence_percent": 100 * confidence if confidence is not None else None,
        "lift": confidence / (cohort_readmissions / cohort_size)
        if confidence is not None and cohort_readmissions else None,
        "small_group": (
            patients < thresholds["minimum_patients"]
            or min(readmissions, no_readmission) < thresholds["minimum_per_class"]
        ),
    }


def compare_rule_conditions(analysis: AnalysisInputs) -> pd.DataFrame:
    """Compare A, B, and their intersection for each fixed condition pair."""
    if not analysis.predictors.index.equals(analysis.target.index):
        raise ValueError("Rule conditions and outcomes must refer to the same rows.")

    questions = analysis.config["questions"]
    q1, q2, q4 = (questions[key] for key in ("Q1", "Q2", "Q4"))
    if q4["lift_reference"] != "full_cohort_readmission_rate":
        raise ValueError("Rule lift requires the full cohort readmission rate.")
    history = prior_visit_groups(analysis)
    no_visits = q1["comparison_groups"][0]
    conditions = {
        "prior_inpatient": analysis.predictors[q1["inpatient_column"]].gt(0),
        "no_recorded_prior_visits": history.eq(no_visits).mask(history.eq(UNKNOWN_HISTORY)),
    }
    stays = stay_groups(analysis.predictors[q2["stay_column"]], q2["stay_groups"])
    stay_labels = {interval["label"] for interval in q2["stay_groups"]}
    positive = analysis.target.eq(CLASS_LABELS[1])
    thresholds = analysis.config["small_groups"]
    records = []

    for rule in q4["rules"]:
        history_condition = rule["history_condition"]
        stay_group = rule["stay_group"]
        if history_condition not in conditions or stay_group not in stay_labels:
            raise ValueError(f"Unknown condition pair in the analysis configuration: {rule!r}.")

        a = conditions[history_condition]
        b = stays.eq(stay_group).mask(stays.eq(MISSING_STAY))
        comparison = {
            "A": _rule_counts(a, positive, thresholds),
            "B": _rule_counts(b, positive, thresholds),
            "A_and_B": _rule_counts(a & b, positive, thresholds),
        }
        for condition, metrics in comparison.items():
            row = {
                "history_condition": history_condition,
                "stay_group": stay_group,
                "condition": condition,
                **metrics,
                "confidence_difference_vs_a_pp": None,
                "confidence_difference_vs_b_pp": None,
            }
            if condition == "A_and_B":
                combined = metrics["confidence_percent"]
                for component in ("A", "B"):
                    confidence = comparison[component]["confidence_percent"]
                    if combined is not None and confidence is not None:
                        row[f"confidence_difference_vs_{component.lower()}_pp"] = combined - confidence
            records.append(row)
    return pd.DataFrame.from_records(records)
