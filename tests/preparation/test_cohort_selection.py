"""Check cohort filtering, representative selection, and aggregate audit output."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

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
        self.rows = []
        for encounter, patient, discharge, label in values:
            row = dict.fromkeys(ENCOUNTER_COLUMNS, "0")
            row.update(
                encounter_id=encounter, patient_nbr=patient,
                discharge_disposition_id=discharge, readmitted=label,
            )
            self.rows.append(row)
        self.rows[0]["race"] = self.rows[2]["race"] = ""
        self.mapping = RawTable(
            Path("IDS_mapping.csv"), "", 0, list(MAPPING_COLUMNS),
            [
                dict(zip(MAPPING_COLUMNS, pair, strict=True))
                for pair in (
                    ("1", "Emergency"), ("", ""),
                    ("discharge_disposition_id", "description"),
                    ("1", "Home"), ("11", "Expired"), ("13", "Hospice"),
                    ("14", "Hospice"), ("18", "NULL"), ("19", "Expired"),
                    ("20", "Expired"), ("21", "Expired"), ("25", "Not Mapped"),
                    ("", ""), ("admission_source_id", "description"),
                )
            ],
        )

    def test_filters_before_selection_and_ranks_numerically(self):
        before = deepcopy(self.rows)
        result = select_cohort(ENCOUNTER_COLUMNS, self.rows, self.mapping)
        reordered = select_cohort(ENCOUNTER_COLUMNS, list(reversed(self.rows)), self.mapping)
        self.assertEqual(result.rows, [before[2], before[4]])
        self.assertEqual(result.rows, reordered.rows)
        self.assertEqual(self.rows, before)
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
            select_cohort(ENCOUNTER_COLUMNS, self.rows + [self.rows[0]], self.mapping)
        self.rows[0]["discharge_disposition_id"] = "97"
        with self.assertRaisesRegex(ValueError, "Unmapped discharge disposition code"):
            select_cohort(ENCOUNTER_COLUMNS, self.rows, self.mapping)

    def test_rejects_invalid_patient_and_original_label(self):
        bad_label = [self.rows[0] | {"readmitted": "OTHER"}]
        with self.assertRaisesRegex(ValueError, "Unexpected target value"):
            select_cohort(ENCOUNTER_COLUMNS, bad_label, self.mapping)
        bad_patient = [self.rows[0] | {"patient_nbr": ""}]
        with self.assertRaisesRegex(ValueError, "Invalid patient_nbr"):
            select_cohort(ENCOUNTER_COLUMNS, bad_patient, self.mapping)

    def test_writes_cohort_and_column_audit(self):
        result = select_cohort(ENCOUNTER_COLUMNS, self.rows, self.mapping)
        cohort_path = self.directory / "cohort.csv"
        audit_path = self.directory / "cohort_audit.csv"
        for _ in range(2):
            save_cohort_result(result, cohort_path=cohort_path, audit_path=audit_path)
        self.assertEqual(read_csv(cohort_path), (list(ENCOUNTER_COLUMNS), [self.rows[2], self.rows[4]]))
        header, audit = read_csv(audit_path)
        self.assertEqual(header, list(AUDIT_COLUMNS))
        self.assertEqual([row["column"] for row in audit], list(ENCOUNTER_COLUMNS))
        race = next(row for row in audit if row["column"] == "race")
        self.assertEqual(
            [race[f"{stage}_empty"] for stage in ("cleaned", "eligible", "cohort")],
            ["2", "1", "1"],
        )
        self.assertEqual(float(race["cohort_missing_rate"]), 0.5)
