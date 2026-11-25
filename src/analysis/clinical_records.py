"""Relate recorded laboratory results and treatment statuses to readmission."""

from dataclasses import dataclass

import pandas as pd

from src.analysis.data import AnalysisInputs
from src.analysis.outcomes import summarize_group_outcomes


UNKNOWN_RECORDING = "unknown"


@dataclass
class ClinicalRecordProfile:
    """The aggregate tables for laboratory coverage, results, and treatment."""

    lab_recording_by_outcome: pd.DataFrame
    lab_results_by_outcome: pd.DataFrame
    treatment_by_outcome: pd.DataFrame


def _lab_recording(analysis: AnalysisInputs, question: dict) -> pd.DataFrame:
    a1c = analysis.predictors["A1Cresult"]
    glucose = analysis.predictors["max_glu_serum"]
    a1c_recorded = a1c.isin(question["result_levels"][a1c.name])
    glucose_recorded = glucose.isin(question["result_levels"][glucose.name])
    no_result = question["no_recorded_result"]
    known = (a1c_recorded | a1c.isin([no_result])) & (glucose_recorded | glucose.isin([no_result]))

    patterns = pd.Series(UNKNOWN_RECORDING, index=a1c.index, name="recording_pattern", dtype="string")
    order = []
    for group in question["recording_groups"]:
        matches = a1c_recorded.eq(group["a1c_recorded"]) & glucose_recorded.eq(group["glucose_recorded"])
        patterns.loc[known & matches] = group["label"]
        order.append(group["label"])
    if (~known).any():
        order.append(UNKNOWN_RECORDING)
    return summarize_group_outcomes(
        patterns.to_frame(), pd.DataFrame({patterns.name: order}), analysis.target, **analysis.config["small_groups"],
    )


def _source_level_outcomes(analysis: AnalysisInputs, column: str, levels: list[str]) -> pd.DataFrame:
    """Retain expected levels and report blank fields and unexpected values."""
    values = analysis.predictors[column].fillna("").rename("value")
    unexpected = sorted(set(values.unique()).difference(levels))
    order = pd.DataFrame({"value": [*levels, *unexpected]})
    summary = summarize_group_outcomes(values.to_frame(), order, analysis.target, **analysis.config["small_groups"])
    status = dict.fromkeys(levels, "recorded") | {"": "blank_field"}
    summary.insert(0, "column", column)
    summary.insert(2, "value_status", summary["value"].map(status).fillna("invalid_value"))
    return summary


def _lab_result_levels(analysis: AnalysisInputs, column: str, levels: list[str], no_result: str) -> pd.DataFrame:
    table = _source_level_outcomes(analysis, column, [*levels, no_result])
    table.loc[table["value"].eq(no_result), "value_status"] = "no_recorded_result"
    recorded = table["value_status"].eq("recorded")
    recorded_patients = int(table.loc[recorded, "patients"].sum())
    table["recorded_result_patients"] = recorded_patients
    table["percent_of_recorded_results"] = 100 * table["patients"].where(recorded) / (recorded_patients or float("nan"))
    return table


def profile_clinical_records(analysis: AnalysisInputs) -> ClinicalRecordProfile:
    """Summarize laboratory recording, result levels, and treatment statuses."""
    question = analysis.config["questions"]["Q3"]
    laboratory = [
        _lab_result_levels(analysis, column, levels, question["no_recorded_result"])
        for column, levels in question["result_levels"].items()
    ]
    treatment = [
        _source_level_outcomes(analysis, column, levels)
        for column, levels in question["treatment_levels"].items()
    ]
    return ClinicalRecordProfile(
        lab_recording_by_outcome=_lab_recording(analysis, question),
        lab_results_by_outcome=pd.concat(laboratory, ignore_index=True),
        treatment_by_outcome=pd.concat(treatment, ignore_index=True),
    )
