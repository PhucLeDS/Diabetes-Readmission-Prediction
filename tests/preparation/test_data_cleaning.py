"""Tests for approved marker replacements and cleaning audit output."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.io_utils import read_csv, write_json
from src.preparation.data_cleaning import AUDIT_COLUMNS, clean_encounters, save_cleaning_result
from src.preparation.data_loading import ENCOUNTER_COLUMNS, RawTable


class CleaningTests(TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(TemporaryDirectory()))
        self.columns = list(ENCOUNTER_COLUMNS)
        first = dict.fromkeys(self.columns, "0")
        first.update(
            encounter_id="001", patient_nbr="010", readmitted=">30",
            race="?", gender="Unknown/Invalid", weight="?",
            payer_code="?", medical_specialty="?", diag_1="?", diag_2="?", diag_3="?",
            A1Cresult="None", max_glu_serum="None",
        )
        second = first.copy()
        second.update(
            encounter_id="002", readmitted="NO", race="", gender="Female", weight="[75-100)",
            payer_code="MC", medical_specialty="Cardiology", diag_1="250.1",
            diag_2="V58", diag_3="401", max_glu_serum="Norm",
        )
        self.table = RawTable(Path("raw.csv"), "", 0, self.columns, [first, second])

    def test_cleans_only_approved_markers_and_keeps_source_records(self):
        before = deepcopy(self.table.rows)
        result = clean_encounters(self.table)
        expected = before[0] | dict.fromkeys(
            ("race", "gender", "weight", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3"), "",
        )
        self.assertEqual(result.rows, [expected, before[1]])
        self.assertEqual(result.columns, tuple(self.columns))
        self.assertEqual(self.table.rows, before)

    def test_audit_covers_every_column_and_distinguishes_markers_from_empty_fields(self):
        result = clean_encounters(self.table)
        self.assertEqual([row["column"] for row in result.audit], self.columns)
        audit = {row["column"]: row for row in result.audit}
        self.assertEqual(audit["race"]["marker_before"], 1)
        self.assertEqual(audit["race"]["marker_after"], 0)
        self.assertEqual(audit["race"]["empty_before"], 1)
        self.assertEqual(audit["race"]["empty_after"], 2)
        self.assertEqual(audit["race"]["missing_rate_before"], 1.0)
        self.assertEqual(audit["race"]["missing_rate_after"], 1.0)
        self.assertEqual(audit["A1Cresult"]["literal_before"], 2)
        self.assertEqual(audit["A1Cresult"]["literal_after"], 2)
        self.assertEqual(audit["A1Cresult"]["empty_after"], 0)
        self.assertEqual(sum(row["changed_cells"] for row in result.audit), 8)

    def test_unapproved_question_marker_is_rejected(self):
        self.table.rows[0]["change"] = "?"
        with self.assertRaisesRegex(ValueError, "Unapproved.*'change'.*row 1"):
            clean_encounters(self.table)

    def test_cleaning_policy_cannot_change_target(self):
        config = self.directory / "preparation.json"
        write_json(config, {"cleaning": {"missing_markers": {"readmitted": "?"}, "preserve_literals": {}}})
        with self.assertRaisesRegex(ValueError, "eligible categorical columns"):
            clean_encounters(self.table, config_path=config)

    def test_saved_csv_and_audit_round_trip_and_retry(self):
        result = clean_encounters(self.table)
        cleaned_path = self.directory / "cleaned.csv"
        audit_path = self.directory / "cleaning_audit.csv"
        for _ in range(2):
            self.assertEqual(
                save_cleaning_result(result, cleaned_path=cleaned_path, audit_path=audit_path),
                (cleaned_path, audit_path),
            )
        header, rows = read_csv(cleaned_path)
        self.assertEqual(header, self.columns)
        self.assertEqual(rows, result.rows)
        header, audit = read_csv(audit_path)
        self.assertEqual(header, list(AUDIT_COLUMNS))
        self.assertEqual(len(audit), 50)
        self.assertEqual(sum(int(row["changed_cells"]) for row in audit), 8)
