"""Check the boundaries and reconciliation of the current encounter tables."""

from unittest import TestCase

import pandas as pd

from src.analysis.current_encounters import profile_current_encounters, stay_groups
from src.analysis.data import ANALYSIS_CONFIG_PATH, AnalysisInputs
from src.io_utils import read_json


class CurrentEncounterTests(TestCase):
    def test_stay_boundaries_and_values_requiring_separate_reporting(self):
        days = pd.Series([0, 1, 3, 4, 7, 8, 14, 15, None], dtype="Int64")
        intervals = read_json(ANALYSIS_CONFIG_PATH)["questions"]["Q2"]["stay_groups"]
        self.assertEqual(stay_groups(days, intervals).tolist(), [
            "outside_configured_range", "1-3", "1-3", "4-7", "4-7", "8-14", "8-14",
            "outside_configured_range", "missing",
        ])
        intervals[1]["minimum"] = 3
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            stay_groups(days, intervals)

    def test_tables_keep_optional_values_in_their_original_history_group(self):
        config = read_json(ANALYSIS_CONFIG_PATH)
        predictors = pd.DataFrame({
            "number_outpatient": pd.array([0, 0, 1, None], dtype="Int64"),
            "number_emergency": pd.array([0, 0, 0, 0], dtype="Int64"),
            "number_inpatient": pd.array([0, 0, 0, 0], dtype="Int64"),
            "time_in_hospital": pd.array([1, None, 15, 8], dtype="Int64"),
            "number_diagnoses": pd.array([1, None, 9, 1], dtype="Int64"),
            "diag_1_group": pd.array(["Circulatory", "Missing", "Unmapped", ""], dtype="string"),
        })
        outcomes = pd.Series(["readmitted", "not_readmitted", "readmitted", "readmitted"], dtype="string")
        profile = profile_current_encounters(AnalysisInputs(predictors, outcomes, config))
        history = config["questions"]["Q1"]["comparison_groups"]

        for table in (profile.stay_by_history, profile.diagnoses_count_by_history, profile.primary_diagnosis_by_history):
            totals = table.groupby("history_group")[["patients", "readmissions"]].sum()
            self.assertEqual(totals.loc[[*history, "unknown"]].values.tolist(), [[2, 1], [1, 1], [1, 1]])

        stays = profile.stay_by_history.set_index(["history_group", "stay_group"])
        self.assertEqual(stays.loc[(history[0], "missing"), "patients"], 1)
        self.assertEqual(stays.loc[(history[1], "outside_configured_range"), "patients"], 1)
        self.assertTrue(pd.isna(stays.loc[(history[0], "8-14"), "readmission_rate_percent"]))

        counts = profile.diagnoses_count_by_history
        self.assertEqual(counts.loc[counts["number_diagnoses"].isna(), "patients"].sum(), 1)
        diagnoses = profile.primary_diagnosis_by_history.groupby("diag_1_group")["patients"].sum()
        self.assertEqual(diagnoses.to_dict(), {"Blank field": 1, "Circulatory": 1, "Missing": 1, "Unmapped": 1})
