"""Check the full prepared dataset contract before CSV export."""

from unittest import TestCase

from src.preparation.data_loading import ENCOUNTER_COLUMNS
from src.preparation.dataset_preparation import (
    CohortResult,
    group_diagnoses,
    select_predictors,
    validate_full_prepared,
)


class FullPreparedTests(TestCase):
    def setUp(self):
        first = dict.fromkeys(ENCOUNTER_COLUMNS, "0")
        first.update(
            encounter_id="10", patient_nbr="101", readmitted="<30", race="",
            diag_1="250.01", diag_2="", diag_3="401", A1Cresult="None",
        )
        second = first | {"encounter_id": "20", "patient_nbr": "102", "readmitted": "NO", "race": "Other"}
        self.cohort = CohortResult(
            columns=ENCOUNTER_COLUMNS, rows=[first, second],
            flow=[], excluded_by_code={}, unclear_retained_by_code={}, audit=[],
        )
        self.selection = select_predictors(group_diagnoses(self.cohort.columns, self.cohort.rows))

    def test_validation_reports_original_labels_and_empty_fields(self):
        statistics = validate_full_prepared(self.selection, self.cohort)
        self.assertEqual((statistics["rows"], statistics["columns"]), (2, 35))
        self.assertEqual(statistics["target_counts"], {"NO": 1, "<30": 1, ">30": 0})
        self.assertEqual(statistics["empty_cells_by_column"]["race"], 1)
        self.assertEqual(statistics["empty_cells_by_column"]["diag_2_group"], 0)
        self.assertEqual(statistics["rows_with_empty_fields"], 1)

    def test_validation_rejects_changed_values_and_grouping(self):
        row = self.selection.rows[0]
        for column, value in (("readmitted", "NO"), ("diag_2_group", "Unmapped"), ("A1Cresult", "")):
            with self.subTest(column=column):
                original = row[column]
                row[column] = value
                with self.assertRaisesRegex(ValueError, f"differs from the cohort.*{column}"):
                    validate_full_prepared(self.selection, self.cohort)
                row[column] = original

    def test_validation_rejects_reordered_rows_and_duplicate_patients(self):
        self.selection.rows.reverse()
        with self.assertRaisesRegex(ValueError, "differs from the cohort.*encounter_id"):
            validate_full_prepared(self.selection, self.cohort)
        self.selection.rows.reverse()
        self.selection.rows[1]["patient_nbr"] = "101"
        with self.assertRaisesRegex(ValueError, "Duplicate prepared patient_nbr"):
            validate_full_prepared(self.selection, self.cohort)
