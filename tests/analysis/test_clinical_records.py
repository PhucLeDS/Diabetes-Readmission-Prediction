"""Check recording patterns and denominators for clinical record tables."""

from unittest import TestCase

import pandas as pd

from src.analysis.clinical_records import profile_clinical_records
from src.analysis.data import ANALYSIS_CONFIG_PATH, AnalysisInputs
from src.io_utils import read_json


class ClinicalRecordTests(TestCase):
    def test_pattern_and_level_tables_keep_none_blank_and_invalid_values_distinct(self):
        config = read_json(ANALYSIS_CONFIG_PATH)
        predictors = pd.DataFrame({
            "A1Cresult": ["None", "Norm", "None", ">8", "", "Unexpected"],
            "max_glu_serum": ["None", "None", ">200", "Norm", ">300", "None"],
            "insulin": ["No", "Up", "No", "Down", "None", "Steady"],
            "change": ["No", "Ch", "No", "Ch", "", "No"],
            "diabetesMed": ["No", "Yes", "No", "Yes", "Yes", "No"],
        }, dtype="string")
        target = pd.Series(["not_readmitted", "readmitted"] * 3, dtype="string")
        profile = profile_clinical_records(AnalysisInputs(predictors, target, config))
        patterns = profile.lab_recording_by_outcome
        self.assertEqual(patterns["recording_pattern"].tolist(), ["neither", "a1c_only", "glucose_only", "both", "unknown"])
        self.assertEqual(patterns["patients"].tolist(), [1, 1, 1, 1, 2])
        self.assertEqual(patterns["readmissions"].tolist(), [0, 1, 0, 1, 1])

        labs = profile.lab_results_by_outcome
        a1c = labs.loc[labs["column"].eq("A1Cresult")].set_index("value")
        self.assertEqual(a1c.loc["None", "value_status"], "no_recorded_result")
        self.assertEqual(a1c.loc["", "value_status"], "blank_field")
        self.assertEqual(a1c.loc["Unexpected", "value_status"], "invalid_value")
        self.assertEqual(a1c["recorded_result_patients"].unique().tolist(), [2])
        self.assertEqual(a1c.loc["Norm", "percent_of_recorded_results"], 50)
        self.assertEqual(a1c.loc["Norm", "readmission_rate_percent"], 100)
        self.assertTrue(pd.isna(a1c.loc["None", "percent_of_recorded_results"]))
        self.assertEqual(a1c.loc["None", "patients"], 2)
        glucose = labs.loc[labs["column"].eq("max_glu_serum")]
        self.assertEqual(glucose["recorded_result_patients"].unique().tolist(), [3])

        treatment = profile.treatment_by_outcome.set_index(["column", "value"])
        self.assertEqual(treatment.loc[("insulin", "None"), "value_status"], "invalid_value")
        self.assertEqual(treatment.loc[("change", ""), "value_status"], "blank_field")
        for table in (labs, treatment.reset_index()):
            totals = table.groupby("column")[["patients", "readmissions"]].sum()
            self.assertTrue(totals["patients"].eq(6).all())
            self.assertTrue(totals["readmissions"].eq(3).all())

    def test_recorded_result_percent_is_undefined_when_no_one_has_a_result(self):
        config = read_json(ANALYSIS_CONFIG_PATH)
        predictors = pd.DataFrame({
            "A1Cresult": ["None"], "max_glu_serum": ["None"],
            "insulin": ["No"], "change": ["No"], "diabetesMed": ["No"],
        }, dtype="string")
        target = pd.Series(["readmitted"], dtype="string")
        profile = profile_clinical_records(AnalysisInputs(predictors, target, config))
        self.assertTrue(profile.lab_results_by_outcome["recorded_result_patients"].eq(0).all())
        self.assertTrue(profile.lab_results_by_outcome["percent_of_recorded_results"].isna().all())
        self.assertEqual(profile.lab_recording_by_outcome["patients"].tolist(), [1, 0, 0, 0])
