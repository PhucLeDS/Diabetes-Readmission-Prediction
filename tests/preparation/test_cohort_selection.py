"""Check cohort filtering, representative selection, and aggregate audit output."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from src.io_utils import read_csv
from src.preparation.data_loading import ENCOUNTER_COLUMNS, MAPPING_COLUMNS, RawTable
from src.preparation.dataset_preparation import AUDIT_COLUMNS, save_cohort_result, select_cohort


class CohortTests(TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(TemporaryDirectory()))
        values = [
            ("1", "100", "11", "NO"),
            ("10", "100", "1", ">30"),
            ("2", "100", "18", "<30"),
            ("4", "200", "13", "NO"),
            ("3", "300", "25", "NO"),
        ]
        rows = []
        for encounter, patient, discharge, label in values:
            row = dict.fromkeys(ENCOUNTER_COLUMNS, "0")
            row.update(
                encounter_id=encounter, patient_nbr=patient,
                discharge_disposition_id=discharge, readmitted=label,
            )
            rows.append(row)
        self.frame = pd.DataFrame(rows, dtype="string")
        self.frame.loc[[0, 2], "race"] = ""
        self.mapping = RawTable(
            Path("IDS_mapping.csv"), "", 0,
            pd.DataFrame(
                (
                    ("1", "Emergency"), ("", ""),
                    ("discharge_disposition_id", "description"),
                    ("1", "Home"), ("11", "Expired"), ("13", "Hospice"),
                    ("14", "Hospice"), ("18", "NULL"), ("19", "Expired"),
                    ("20", "Expired"), ("21", "Expired"), ("25", "Not Mapped"),
                    ("", ""), ("admission_source_id", "description"),
                ),
                columns=MAPPING_COLUMNS, dtype="string",
            ),
        )

    def test_filters_before_selection_and_ranks_numerically(self):
        before = self.frame.copy()
        result = select_cohort(self.frame, self.mapping)
        reordered = select_cohort(self.frame.iloc[::-1], self.mapping)
        pd.testing.assert_frame_equal(result.frame, before.iloc[[2, 4]].reset_index(drop=True))
        pd.testing.assert_frame_equal(result.frame, reordered.frame)
        pd.testing.assert_frame_equal(self.frame, before)
        self.assertEqual(result.excluded_by_code, {"11": 1, "13": 1, "14": 0, "19": 0, "20": 0, "21": 0})
        self.assertEqual(result.unclear_retained_by_code, {"18": 1, "25": 1})
        self.assertEqual(
            [(row["encounters"], row["patients"], row["removed"]) for row in result.flow],
            [(5, 3, 0), (3, 2, 2), (2, 2, 1)],
        )
        self.assertEqual(
            [result.flow[-1][label] for label in ("NO", "<30", ">30")], [1, 1, 0],
        )

    def test_rejects_duplicate_encounter_and_unmapped_discharge(self):
        with self.assertRaisesRegex(ValueError, "Duplicate encounter_id"):
            select_cohort(self.frame.iloc[[0, 0]], self.mapping)
        self.frame.at[0, "discharge_disposition_id"] = "97"
        with self.assertRaisesRegex(ValueError, "Unmapped discharge disposition code"):
            select_cohort(self.frame, self.mapping)

    def test_rejects_invalid_patient_and_original_label(self):
        bad_label = self.frame.iloc[:1].assign(readmitted="OTHER")
        with self.assertRaisesRegex(ValueError, "Unexpected target value"):
            select_cohort(bad_label, self.mapping)
        bad_patient = self.frame.iloc[:1].assign(patient_nbr="")
        with self.assertRaisesRegex(ValueError, "Invalid patient_nbr"):
            select_cohort(bad_patient, self.mapping)

    def test_numeric_order_preserves_large_ids_and_breaks_ties_by_original_text(self):
        frame = self.frame.iloc[[1, 1, 1, 1]].reset_index(drop=True)
        frame["encounter_id"] = pd.array([
            "018446744073709551617", "18446744073709551617", "18446744073709551616", "10",
        ], dtype="string")
        frame["patient_nbr"] = pd.array(["100", "100", "100", "200"], dtype="string")
        result = select_cohort(frame, self.mapping)
        self.assertEqual(result.frame["encounter_id"].tolist(), ["10", "18446744073709551616"])

        tied = select_cohort(frame.iloc[[1, 0, 3]], self.mapping)
        self.assertEqual(tied.frame["encounter_id"].tolist(), ["10", "018446744073709551617"])

    def test_writes_cohort_and_column_audit(self):
        result = select_cohort(self.frame, self.mapping)
        cohort_path = self.directory / "cohort.csv"
        audit_path = self.directory / "cohort_audit.csv"
        for _ in range(2):
            save_cohort_result(result, cohort_path=cohort_path, audit_path=audit_path)
        pd.testing.assert_frame_equal(
            read_csv(cohort_path, expected_columns=ENCOUNTER_COLUMNS),
            self.frame.iloc[[2, 4]].reset_index(drop=True),
        )
        audit = read_csv(audit_path, expected_columns=AUDIT_COLUMNS)
        self.assertEqual(audit["column"].tolist(), list(ENCOUNTER_COLUMNS))
        race = audit.set_index("column").loc["race"]
        self.assertEqual(
            [race[f"{stage}_empty"] for stage in ("cleaned", "eligible", "cohort")],
            ["2", "1", "1"],
        )
        self.assertEqual(float(race["cohort_missing_rate"]), 0.5)
