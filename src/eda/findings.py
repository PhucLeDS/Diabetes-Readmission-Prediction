"""Build a concise handoff from the EDA evidence tables."""

import pandas as pd

from src.eda.data import EdaInputs
from src.eda.descriptive import CompletenessProfile, NumericProfile
from src.eda.nominal import NominalProfile
from src.eda.relationships import RelationshipProfile


def summarize_findings(
    eda: EdaInputs,
    completeness: CompletenessProfile,
    numeric: NumericProfile,
    nominal: NominalProfile,
    relationships: RelationshipProfile,
) -> pd.DataFrame:
    """Describe the cohort and key findings using the same values as the exported tables."""
    total = len(eda.cohort)
    coverage = completeness.lab_recording_patterns.set_index(
        ["a1c_result_recorded", "glucose_result_recorded"],
    )
    neither = coverage.at[(False, False), "patients"]
    prior = numeric.prior_visit_coverage.set_index("service_types_with_prior_visits")
    zero_prior = prior.at[0, "patients"]
    stay = relationships.stay_service_summary
    medication_by_stay = stay.loc[stay["measure"].eq("num_medications")].dropna(subset=["time_in_hospital"])
    shortest, longest = medication_by_stay.iloc[0], medication_by_stay.iloc[-1]
    medication = nominal.medication_status.set_index("column")
    diagnoses = nominal.diagnosis_summary.set_index("group")
    outcome = relationships.outcome_distribution.set_index(["representation", "label"])
    readmitted = outcome.at[("Any Readmission", "readmitted"), "patients"]

    rows = [
        (
            "Cohort selection",
            f"{total:,} patients with one representative encounter each. "
            f"{eda.unclear_discharge_count:,} encounters have unclear discharge descriptions.",
            "The patient is the unit of the EDA summaries.",
            "Encounter IDs establish the selection rule. Clinical chronology remains unverified.",
            "cohort_flow.csv, eda_manifest.json",
        ),
        (
            "Empty fields",
            f"{completeness.empty_field_count:,} empty fields across "
            f"{completeness.patients_with_empty_fields:,} patients.",
            "Field counts measure missing cells. Patient counts account for overlap across columns.",
            "Special source categories require separate counts.",
            "missing_summary.csv, missing_patterns.csv",
        ),
        (
            "Laboratory coverage",
            f"{neither:,}/{total:,} patients ({100 * neither / total:.2f}%) have neither result recorded.",
            "Distributions of result levels apply to patients with a result for the relevant test.",
            "How does test recording vary across clinically interpretable groups?",
            "lab_recording_patterns.csv, laboratory_results.csv",
        ),
        (
            "Prior service use",
            f"{zero_prior:,}/{total:,} patients ({100 * zero_prior / total:.2f}%) have zero visits "
            "recorded across all three service types.",
            "The positive-count distributions describe prior use among patients with a recorded visit.",
            "The counts cover the year preceding the representative encounter.",
            "prior_visit_coverage.csv, prior_visit_positive_profile.csv",
        ),
        (
            "Service use by stay length",
            f"Median number of medications by stay length: {shortest['median']:g} for "
            f"{shortest['time_in_hospital']:g} day(s) (n = {shortest['patients']:,}) and "
            f"{longest['median']:g} for {longest['time_in_hospital']:g} days (n = {longest['patients']:,}).",
            "The stay-specific quartiles describe the spread within each group.",
            "Both measures describe the same encounter. Their relationship is descriptive.",
            "stay_service_summary.csv",
        ),
        (
            "Medication status",
            f"Chlorpropamide Down: {medication.at['chlorpropamide', 'down_patients']:,} of "
            f"{total:,} patients.",
            "Exact counts preserve the scale of rare levels.",
            "The matrix covers the 12 retained medication fields.",
            "medication_status.csv",
        ),
        (
            "Diagnosis groups",
            f"{diagnoses.at['Circulatory', 'patients_at_any_position']:,} patients have Circulatory "
            "at one or more positions.",
            "A patient contributes once to each group in the any-position summary.",
            "How much source-code detail is combined within the broad groups?",
            "diagnosis_summary.csv",
        ),
        (
            "Any Readmission",
            f"{readmitted:,}/{total:,} patients ({100 * readmitted / total:.2f}%) have a recorded readmission.",
            "The binary target combines the source timing categories <30 and >30.",
            "NO refers to the source's observation coverage. The exact interval is unavailable.",
            "outcome_distribution.csv",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["finding", "evidence", "interpretation", "limit_or_open_question", "source_table"],
    )
