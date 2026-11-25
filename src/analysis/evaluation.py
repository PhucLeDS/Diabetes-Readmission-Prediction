"""Build a Weka evaluation reference from the cohort findings."""

import pandas as pd

from src.analysis.clinical_records import ClinicalRecordProfile
from src.analysis.current_encounters import CurrentEncounterProfile
from src.analysis.data import AnalysisInputs
from src.analysis.prior_visits import PriorVisitProfile


def _format_rate(value: float | None) -> str:
    return f"{value:.2f}%" if pd.notna(value) else "undefined"


def build_evaluation_map(
    analysis: AnalysisInputs,
    prior_visits: PriorVisitProfile,
    current_encounters: CurrentEncounterProfile,
    clinical_records: ClinicalRecordProfile,
    rule_comparisons: pd.DataFrame,
) -> pd.DataFrame:
    """Build one evidence and evaluation record per analysis question."""
    questions = analysis.config["questions"]
    alert = prior_visits.history_rule_confusion_matrix[
        ["true_positives", "false_negatives", "false_positives"]
    ].iloc[0]
    q2 = questions["Q2"]
    stay_table = current_encounters.stay_by_history.set_index(["history_group", "stay_group"])
    short_band, long_band = q2["stay_groups"][0]["label"], q2["stay_groups"][-1]["label"]
    short = stay_table.loc[(q2["focus_group"], short_band)]
    long = stay_table.loc[(q2["focus_group"], long_band)]
    patterns = clinical_records.lab_recording_by_outcome.set_index("recording_pattern")
    neither = patterns.loc[questions["Q3"]["recording_reference_group"]]
    both = patterns.loc["both"]

    rules = rule_comparisons.set_index(["history_condition", "stay_group", "condition"])
    combined_findings = []
    for rule in questions["Q4"]["rules"]:
        history, stay = rule["history_condition"], rule["stay_group"]
        combined = rules.loc[(history, stay, "A_and_B")]
        a = rules.loc[(history, stay, "A")]
        b = rules.loc[(history, stay, "B")]
        description = questions["Q4"]["history_condition_definitions"][history].rstrip(".")
        combined_findings.append(
            f"History condition ({description}) and stay of {stay} days gave {_format_rate(combined['confidence_percent'])} confidence "
            f"({combined['matched_readmissions']:,}/{combined['matched_patients']:,}), compared with "
            f"{_format_rate(a['confidence_percent'])} for history alone and "
            f"{_format_rate(b['confidence_percent'])} for the stay band alone."
        )

    evidence = {
        "Q1": (
            f"The history alert captured {alert['true_positives']:,} recorded readmissions and missed "
            f"{alert['false_negatives']:,}, with {alert['false_positives']:,} false positives.",
            "The alert uses complete histories and counts visits recorded in the previous year.",
            "Precision, recall, alerts, false positives, and missed cases with zero recorded visits.",
            "prior_visit_summary.csv, history_rule_confusion_matrix.csv",
        ),
        "Q2": (
            f"With no recorded prior visits, the rate was {_format_rate(short['readmission_rate_percent'])} "
            f"({short['readmissions']:,}/{short['patients']:,}) for {short_band} days and "
            f"{_format_rate(long['readmission_rate_percent'])} "
            f"({long['readmissions']:,}/{long['patients']:,}) for {long_band} days.",
            "The cohort contrasts are unadjusted. Encounter fields describe information accrued during the stay.",
            "Recall and false-positive counts among patients with zero recorded prior visits.",
            "stay_by_history.csv, diagnoses_count_by_history.csv, primary_diagnosis_by_history.csv",
        ),
        "Q3": (
            f"Neither result was recorded for {neither['patients']:,}/{neither['cohort_size']:,} patients. "
            f"Both results were recorded for {both['patients']:,}, including {both['readmissions']:,} "
            "recorded readmissions.",
            "Recording patterns reflect source coverage. Treatment statuses describe associations.",
            "Rule coverage, matched readmissions, and model precision and recall by recording pattern.",
            "lab_recording_by_outcome.csv, lab_results_by_outcome.csv, treatment_by_outcome.csv",
        ),
        "Q4": (
            " ".join(combined_findings),
            "The conditions are prespecified, and the rates and lift use the full historical cohort.",
            "Matched patients and readmissions, coverage, support, confidence, and lift.",
            "rule_comparisons.csv",
        ),
    }
    return pd.DataFrame.from_records([
        {
            "question_id": question_id,
            "question": questions[question_id]["question"],
            "finding": finding,
            "limitation": limitation,
            "weka_comparison": questions[question_id]["evaluation"],
            "metrics": metrics,
            "evaluation_dataset": "Validation, following development on Train",
            "source_tables": source_tables,
        }
        for question_id, (finding, limitation, metrics, source_tables) in evidence.items()
    ])
