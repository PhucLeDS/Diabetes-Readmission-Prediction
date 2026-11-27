"""Build a concise handoff from the EDA evidence tables."""

import pandas as pd

from src.eda.admission_context import AdmissionContext
from src.eda.cohort_selection import CohortSelectionProfile
from src.eda.data import EdaInputs
from src.eda.descriptive import CompletenessProfile, NumericProfile
from src.eda.diagnosis_detail import DiagnosisDetail
from src.eda.information_coverage import InformationCoverage
from src.eda.medication_detail import MedicationDetail
from src.eda.nominal import NominalProfile
from src.eda.relationships import RelationshipProfile


def _share(patients: int, denominator: int) -> str:
    if denominator == 0:
        return f"{patients:,}/{denominator:,} (percentage unavailable)"
    return f"{patients:,}/{denominator:,} ({100 * patients / denominator:.2f}%)"


def summarize_findings(
    eda: EdaInputs,
    completeness: CompletenessProfile,
    numeric: NumericProfile,
    nominal: NominalProfile,
    relationships: RelationshipProfile,
    *,
    selection: CohortSelectionProfile,
    information_coverage: InformationCoverage,
    admission: AdmissionContext,
    diagnosis_detail: DiagnosisDetail,
    medication_detail: MedicationDetail,
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

    overlap = selection.discharge_patient_overlap.set_index("group")
    eligible_visits = selection.eligible_encounters_per_patient
    multiple_eligible = eligible_visits.loc[eligible_visits["eligible_encounters_per_patient"].gt(1), "patients"].sum()
    lab_history = information_coverage.lab_recording_by_prior_history.set_index(
        ["prior_history", "lab_recording_pattern"],
    )
    no_history = lab_history.loc[("no_recorded_prior_visits", "neither")]
    with_history = lab_history.loc[("any_recorded_prior_visit", "neither")]
    lab_subgroups = information_coverage.lab_recording_by_characteristic
    both_by_type = lab_subgroups.loc[
        lab_subgroups["characteristic"].eq("admission_type_id") & lab_subgroups["lab_recording_pattern"].eq("both")
    ].set_index("value")
    both_at_code_six = int(both_by_type["patients"].get("6", 0))
    code_six_patients = int(both_by_type["group_patients"].get("6", 0))
    both_patients = int(both_by_type["patients"].sum())
    type_source = admission.admission_type_source
    emergency = type_source.loc[type_source["admission_type_code"].eq("1")]
    emergency_room = emergency.loc[emergency["admission_source_code"].eq("7"), "patients"].sum()
    diagnosis_freq = numeric.numeric_frequency.loc[numeric.numeric_frequency["column"].eq("number_diagnoses")]
    nine = diagnosis_freq.loc[diagnosis_freq["value"].eq(9), "patients"].sum()
    above_nine = diagnosis_freq.loc[diagnosis_freq["value"].gt(9), "patients"].sum()
    diabetes_gain = diagnosis_detail.diagnosis_group_gain.set_index("group").loc["Diabetes"]
    circulatory_codes = diagnosis_detail.diagnosis_source_code_summary.set_index("group").at[
        "Circulatory", "distinct_source_codes",
    ]
    insulin_metformin = medication_detail.insulin_medication_pairs.set_index("medication").loc["metformin"]
    reconciled = medication_detail.medication_flag_reconciliation.set_index(["source_flag", "excluded_status"])
    diabetes_flag = reconciled.loc[("diabetesMed", "matching_status_recorded")]
    change_flag = reconciled.loc[("change", "matching_status_recorded")]

    eligible_age = selection.discharge_eligibility_profile.set_index(["group", "attribute", "value"])
    older_eligible = eligible_age.loc[("eligible", "age", "[80-90)")]
    older_excluded = eligible_age.loc[("excluded", "age", "[80-90)")]
    depth = diagnosis_detail.diagnosis_position_depth
    mapped = depth.loc[depth["mapped_positions"].eq(3)]
    repeated = mapped.loc[mapped["distinct_mapped_groups"].lt(3), "patients"].sum()
    pairs = diagnosis_detail.diagnosis_group_pairs.set_index(["group_1", "group_2"])
    diabetes_circulatory = pairs["patients"].get(("Circulatory", "Diabetes"), 0)
    drug_counts = medication_detail.medication_count_distribution
    multiple_drugs = drug_counts.loc[
        drug_counts["measure"].eq("recorded_status") & drug_counts["medications"].ge(2), "patients"
    ].sum()

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
            "Discharge eligibility",
            f"{overlap.at['eligible_and_excluded', 'patients']:,} patients have both eligible and excluded "
            f"encounters, and {overlap.at['excluded_only', 'patients']:,} have only excluded encounters. "
            f"Ages 80 to under 90 account for {_share(older_excluded['encounters'], older_excluded['group_encounters'])} "
            f"of excluded encounters and {_share(older_eligible['encounters'], older_eligible['group_encounters'])} "
            "of eligible encounters.",
            "A patient can contribute an eligible representative encounter even if another encounter was excluded.",
            "The overlap counts patients. The two age percentages count encounters and can include repeat visits.",
            "discharge_patient_overlap.csv, discharge_eligibility_profile.csv",
        ),
        (
            "Multiple eligible encounters",
            f"{_share(multiple_eligible, total)} patients have more than one eligible encounter.",
            "The representative encounter rule limits contributions to one encounter per eligible patient.",
            "Encounter identifiers alone cannot establish clinical chronology.",
            "eligible_encounters_per_patient.csv, representative_encounter_profile.csv",
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
            f"{_share(neither, total)} patients have neither result recorded.",
            "Distributions of result levels apply to patients with a result for the relevant test.",
            "The source records result availability. Whether the tests were performed is unknown.",
            "lab_recording_patterns.csv, laboratory_results.csv",
        ),
        (
            "Laboratory coverage by prior visits",
            f"Neither result is recorded for {_share(no_history['patients'], no_history['group_patients'])} "
            f"patients with zero recorded prior visits and {_share(with_history['patients'], with_history['group_patients'])} "
            "patients with at least one recorded prior visit.",
            "The denominators are the two prior service history groups.",
            "Recorded visits cover outpatient, emergency, and inpatient services in the prior year.",
            "lab_recording_by_prior_history.csv",
        ),
        (
            "Laboratory coverage by admission type",
            f"Admission type code 6 (NULL) accounts for {_share(both_at_code_six, both_patients)} patients "
            f"with both results recorded. Within code 6, {_share(both_at_code_six, code_six_patients)} "
            "patients have both results recorded.",
            "The two percentages quantify concentration among patients with both results and coverage within the code.",
            "Code 6 has an unclear lookup description.",
            "lab_recording_by_characteristic.csv, admission_lookup.csv",
        ),
        (
            "Admission type and source",
            f"The Emergency Room is the recorded source for {_share(emergency_room, emergency['patients'].sum())} "
            "Emergency admissions.",
            "The source percentage is conditional on admission type.",
            "Admission codes with unclear descriptions and groups with few patients require separate interpretation.",
            "admission_type_source.csv",
        ),
        (
            "Prior service use",
            f"{_share(zero_prior, total)} patients have zero visits "
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
            "Diagnosis count distribution",
            f"{_share(nine, total)} patients have exactly nine diagnoses recorded. "
            f"{_share(above_nine, total)} patients have more than nine.",
            "The distribution is concentrated at nine and the groups above nine contain few patients.",
            "The source provides no explanation for the concentration at nine.",
            "numeric_frequency.csv",
        ),
        (
            "Diagnosis groups",
            f"{diagnoses.at['Circulatory', 'patients_at_any_position']:,} patients have Circulatory "
            f"at one or more positions. The group contains {circulatory_codes:,} distinct source codes.",
            "A patient contributes once to each group in the any-position summary.",
            "A broad diagnosis group can combine clinically distinct source codes.",
            "diagnosis_summary.csv, diagnosis_source_code_summary.csv",
        ),
        (
            "Diabetes diagnosis group coverage",
            f"The Diabetes group increases from {diabetes_gain['patients_at_primary_position']:,} patients at the "
            f"primary position to {diabetes_gain['patients_at_any_position']:,} patients at any position. The additional "
            f"{diabetes_gain['additional_patients']:,} patients represent "
            f"{diabetes_gain['additional_percent_of_cohort']:.2f} percentage points of the cohort.",
            "This gain measures coverage of the Diabetes code group within an existing diabetes cohort.",
            "The definition covers three recorded diagnosis positions at the representative encounter.",
            "diagnosis_group_gain.csv",
        ),
        (
            "Diagnosis depth and overlap",
            f"Among patients with three mapped positions, {_share(repeated, mapped['patients'].sum())} "
            f"have a repeated diagnosis group. Circulatory and Diabetes occur together in "
            f"{_share(diabetes_circulatory, total)} patients across the full cohort.",
            "Repeated groups and co-occurring groups describe different aspects of the same encounter.",
            "Each pair counts a patient once. Patients may contribute to several pairs.",
            "diagnosis_position_depth.csv, diagnosis_group_pairs.csv",
        ),
        (
            "Insulin and metformin records",
            f"{_share(multiple_drugs, total)} patients have a recorded status of Steady, Up, or Down for two or "
            f"more retained medications. Metformin is recorded alongside insulin for "
            f"{_share(insulin_metformin['pair_patients'], insulin_metformin['insulin_patients'])} "
            "patients with recorded insulin.",
            "Patients can contribute to more than one insulin and medication pair.",
            "The recorded statuses give no information about the timing or sequence of treatment.",
            "insulin_medication_pairs.csv, medication_count_distribution.csv",
        ),
        (
            "Medication flag reconciliation",
            f"Excluded fields contain a matching medication status for "
            f"{_share(diabetes_flag['patients'], diabetes_flag['comparison_patients'])} patients with diabetesMed = Yes "
            "and no recorded non-No status in the retained fields. A matching Up or Down status appears for "
            f"{_share(change_flag['patients'], change_flag['comparison_patients'])} patients with change = Ch "
            "and no retained Up or Down status.",
            "Each denominator is the subgroup without the specified retained medication status.",
            "The comparisons cannot establish why the flags were set.",
            "medication_flag_reconciliation.csv, medication_flags.csv",
        ),
        (
            "Any Readmission",
            f"{_share(readmitted, total)} patients have a recorded readmission.",
            "The binary target combines the source timing categories <30 and >30.",
            "NO refers to the source's observation coverage. The exact interval is unavailable.",
            "outcome_distribution.csv",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["finding", "evidence", "interpretation", "limitation", "source_table"],
    )


def analysis_directions() -> pd.DataFrame:
    """Connect descriptive findings to the comparisons needed for an integrated readmission analysis."""
    rows = [
        (
            "Cohort composition and outcome coverage",
            "Cohort selection, Discharge eligibility, Multiple eligible encounters, Empty fields, Any Readmission",
            "Establish the patient-level baseline. Trace how discharge exclusions and representative encounter "
            "selection shape age composition. Describe patients with unclear discharge information in the source cohort.",
            "Use encounter denominators for selection effects. Use eligible patients and recorded readmissions "
            "for the baseline and subgroup outcome counts.",
            "Establish the population to which all subsequent comparisons apply and identify gaps in documentation.",
        ),
        (
            "Prior service use and admission context",
            "Prior service use, Admission type and source, Service use by stay length",
            "Compare readmission across prior inpatient, emergency and outpatient use. Examine the mix of ages "
            "and admission sources, then compare stay length and encounter services within common admission types.",
            "For each group report readmissions per patient count and its share of all recorded readmissions. "
            "Show patients with zero prior visits and retain unclear admission codes.",
            "Distinguish groups with a high readmission rate from groups accounting for many cases. Examine whether "
            "the same contrasts persist within admission contexts.",
        ),
        (
            "Laboratory recording and treatment documentation",
            "Laboratory coverage, Laboratory coverage by prior visits, Laboratory coverage by admission type, "
            "Medication status, Insulin and metformin records, Medication flag reconciliation",
            "Compare lab recording patterns across prior use and admission type. Within those contexts, examine "
            "recorded results and medication combinations, then reconcile retained medication fields with source flags.",
            "Show recording coverage in each group before comparing result levels among patients with that result. "
            "For medication pairs, use patients with recorded insulin and show small or unknown groups separately.",
            "Identify which outcome contrasts have adequate documentation. Retain limits from selective recording, "
            "sparse medication statuses and the absence of treatment timing.",
        ),
        (
            "Diagnosis burden and coding coverage",
            "Diagnosis count distribution, Diagnosis groups, Diabetes diagnosis group coverage, Diagnosis depth and overlap",
            "Compare outcome counts by diagnosis count, position of the Diabetes group and common co-occurring groups. "
            "Examine which contrasts persist within comparable prior-use or admission groups.",
            "Count each patient once within each comparison. Distinguish primary, secondary-only and absent Diabetes "
            "codes. Show missing positions, repeated groups and sparse counts above nine.",
            "Assess how the choice of diagnosis representation affects the apparent readmission pattern and identify "
            "where overlapping groups preclude adding case counts.",
        ),
    ]
    return pd.DataFrame(rows, columns=[
        "direction", "supporting_findings", "analysis_scope", "denominators_and_checks", "contribution_to_synthesis",
    ])
