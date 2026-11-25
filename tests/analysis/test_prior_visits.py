"""Check Q1 reconciliation and the treatment of incomplete service histories."""

from unittest import TestCase

import pandas as pd

from src.analysis.data import ANALYSIS_CONFIG_PATH, AnalysisInputs
from src.analysis.prior_visits import prior_visit_groups, profile_prior_visits
from src.io_utils import read_json


class PriorVisitTests(TestCase):
    def inputs(self, visits, outcomes):
        config = read_json(ANALYSIS_CONFIG_PATH)
        predictors = pd.DataFrame(visits, columns=config["questions"]["Q1"]["visit_columns"], dtype="Int64")
        return AnalysisInputs(predictors, pd.Series(outcomes, dtype="string"), config)

    def test_missing_history_keeps_unknown_counts_out_of_the_rule_metrics(self):
        analysis = self.inputs(
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 1, 1), (1, None, 0), (0, 0, None)],
            ["readmitted", "readmitted", "not_readmitted", "readmitted", "readmitted", "not_readmitted"],
        )
        profile = profile_prior_visits(analysis)
        self.assertEqual(prior_visit_groups(analysis).tolist(), [
            "no_recorded_prior_visits", "any_recorded_prior_visit", "any_recorded_prior_visit",
            "any_recorded_prior_visit", "unknown", "unknown",
        ])
        self.assertEqual(profile.prior_visit_summary["patients"].tolist(), [1, 3, 2])
        self.assertEqual(profile.prior_visit_summary["readmissions"].tolist(), [1, 2, 1])
        self.assertEqual(profile.prior_visit_summary["share_of_readmissions_percent"].tolist(), [25, 50, 25])
        self.assertTrue(profile.prior_visit_summary["small_group"].all())

        combinations = profile.prior_visit_combinations
        self.assertEqual(combinations["patients"].tolist(), [1, 2, 0, 0, 0, 0, 0, 1, 2])
        self.assertEqual(combinations["readmissions"].sum(), 4)
        self.assertEqual(profile.prior_inpatient_levels["patients"].tolist(), [2, 2, 1, 1])

        rule = profile.history_rule_confusion_matrix.iloc[0]
        self.assertEqual(rule["evaluated_patients"], 4)
        self.assertEqual(rule["unknown_history_patients"], 2)
        self.assertEqual(rule["unknown_history_readmissions"], 1)
        self.assertEqual(
            tuple(rule[field] for field in ("true_positives", "false_positives", "false_negatives", "true_negatives")),
            (2, 1, 1, 0),
        )
        self.assertAlmostEqual(rule["precision_percent"], 100 * 2 / 3)
        self.assertAlmostEqual(rule["recall_percent"], 100 * 2 / 3)

    def test_zero_denominators_are_missing_and_empty_combinations_remain(self):
        analysis = self.inputs([(0, 0, 0)], ["not_readmitted"])
        profile = profile_prior_visits(analysis)
        self.assertEqual(len(profile.prior_visit_combinations), 8)
        self.assertTrue(profile.prior_visit_summary["share_of_readmissions_percent"].isna().all())
        self.assertTrue(pd.isna(profile.prior_visit_summary.iloc[1]["readmission_rate_percent"]))
        rule = profile.history_rule_confusion_matrix.iloc[0]
        self.assertEqual(rule["true_negatives"], 1)
        self.assertTrue(pd.isna(rule["precision_percent"]))
        self.assertTrue(pd.isna(rule["recall_percent"]))
