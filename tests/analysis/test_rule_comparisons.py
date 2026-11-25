"""Check condition counts, denominators, and undefined rule metrics."""

from unittest import TestCase

import pandas as pd

from src.analysis.data import ANALYSIS_CONFIG_PATH, AnalysisInputs
from src.analysis.rule_comparisons import compare_rule_conditions
from src.io_utils import read_json


class RuleComparisonTests(TestCase):
    def summarize(self, records, outcomes):
        config = read_json(ANALYSIS_CONFIG_PATH)
        predictors = pd.DataFrame(records, columns=[
            "number_outpatient", "number_emergency", "number_inpatient", "time_in_hospital",
        ], dtype="Int64")
        target = pd.Series(outcomes, dtype="string")
        result = compare_rule_conditions(AnalysisInputs(predictors, target, config))
        return result.set_index(["history_condition", "stay_group", "condition"])

    def test_confirmed_matches_and_missing_conditions(self):
        rules = self.summarize(
            [(0, 0, 1, 8), (0, 0, 1, 3), (0, 0, 0, 14), (0, 0, 0, None), (0, 0, None, 9)],
            ["readmitted", "not_readmitted", "not_readmitted", "readmitted", "readmitted"],
        )
        first = rules.xs(("prior_inpatient", "8-14"), level=("history_condition", "stay_group"))
        self.assertEqual(first["matched_patients"].tolist(), [2, 3, 1])
        self.assertEqual(first["matched_readmissions"].tolist(), [1, 2, 1])
        self.assertEqual(first["unknown_condition_patients"].tolist(), [1, 1, 1])
        self.assertEqual(first["coverage_percent"].tolist(), [40, 60, 20])
        self.assertEqual(first["support_percent"].tolist(), [20, 40, 20])
        self.assertAlmostEqual(first.loc["A_and_B", "lift"], 1 / (3 / 5))
        self.assertAlmostEqual(first.loc["A_and_B", "confidence_difference_vs_a_pp"], 50)
        self.assertAlmostEqual(first.loc["A_and_B", "confidence_difference_vs_b_pp"], 100 - 100 * 2 / 3)
        self.assertEqual(rules.loc[("no_recorded_prior_visits", "8-14", "A_and_B"), "unknown_condition_patients"], 2)

    def test_no_matches_and_no_positive_outcomes(self):
        rules = self.summarize([(0, 0, 0, 8)], ["not_readmitted"])
        first = rules.xs(("prior_inpatient", "8-14"), level=("history_condition", "stay_group"))
        self.assertEqual(first.loc["A_and_B", "support_percent"], 0)
        self.assertTrue(pd.isna(first.loc["A_and_B", "confidence_percent"]))
        self.assertTrue(pd.isna(first.loc["A_and_B", "confidence_difference_vs_b_pp"]))
        self.assertTrue(rules["lift"].isna().all())
        self.assertEqual(first.loc["B", "confidence_percent"], 0)
