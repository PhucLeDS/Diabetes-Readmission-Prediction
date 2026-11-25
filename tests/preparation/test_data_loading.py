"""Tests for the raw input contract and literal CSV representation."""

import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from src.io_utils import write_csv
from src.preparation import data_loading


class RawLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def make_encounter(self, encounter_id="001", patient_nbr="007", label="NO"):
        row = dict.fromkeys(data_loading.ENCOUNTER_COLUMNS, "")
        row.update(dict.fromkeys(data_loading.COUNT_COLUMNS, "0"))
        row.update(
            encounter_id=encounter_id, patient_nbr=patient_nbr, readmitted=label,
            race="?", gender="Unknown/Invalid", A1Cresult="None",
        )
        return row

    def load_fixture(self, rows, *, columns=data_loading.ENCOUNTER_COLUMNS, expected_rows=None):
        path = self.directory / "diabetic_data.csv"
        write_csv(path, columns, rows, overwrite=True)
        contents = path.read_bytes()
        checksum = sha256(contents).hexdigest()
        with (
            patch.object(data_loading, "ENCOUNTER_SHA256", checksum),
            patch.object(data_loading, "ENCOUNTER_ROWS", len(rows) if expected_rows is None else expected_rows),
        ):
            table = data_loading.load_raw_encounters(path)
        self.assertEqual(path.read_bytes(), contents)
        self.assertEqual(table.sha256, checksum)
        return table

    def test_load_preserves_raw_values_and_repeated_patient_ids(self):
        rows = [self.make_encounter(), self.make_encounter("002", label="<30")]
        table = self.load_fixture(rows)
        self.assertEqual(table.rows, rows)
        self.assertEqual(data_loading.summarize_raw_encounters(table), {
            "encounters": 2,
            "columns": 50,
            "patients": 1,
            "patients_with_multiple_encounters": 1,
            "target_counts": {"NO": 1, "<30": 1, ">30": 0},
        })
        profile = {row["column"]: row for row in data_loading.profile_raw_columns(table)}
        self.assertEqual(profile["A1Cresult"]["None"], 2)
        self.assertEqual(profile["race"]["?"], 2)
        self.assertEqual(profile["gender"]["Unknown/Invalid"], 2)
        self.assertEqual(profile["weight"][""], 2)

    def test_checksum_mismatch_is_rejected_for_both_files(self):
        path = self.directory / "altered.csv"
        path.write_bytes(b"changed")
        for loader in (data_loading.load_raw_encounters, data_loading.load_ids_mapping):
            with self.subTest(loader=loader.__name__):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    loader(path)

    def test_schema_and_size_must_match(self):
        rows = [self.make_encounter()]
        with self.assertRaisesRegex(ValueError, "column order"):
            self.load_fixture(rows, columns=tuple(reversed(data_loading.ENCOUNTER_COLUMNS)))
        with self.assertRaisesRegex(ValueError, "expected 2 data rows, got 1"):
            self.load_fixture(rows, expected_rows=2)

    def test_invalid_ids_counts_and_labels_are_rejected(self):
        cases = (
            ("encounter_id", "-", "column 'encounter_id'"),
            ("patient_nbr", "", "column 'patient_nbr'"),
            ("time_in_hospital", "2.5", "column 'time_in_hospital'"),
            ("readmitted", "UNKNOWN", "Unexpected target value"),
        )
        for column, value, error in cases:
            with self.subTest(column=column, value=value):
                row = self.make_encounter()
                row[column] = value
                with self.assertRaisesRegex(ValueError, error):
                    self.load_fixture([row])
        with self.assertRaisesRegex(ValueError, "Duplicate encounter_id"):
            self.load_fixture([self.make_encounter(), self.make_encounter(patient_nbr="008")])

    def test_mapping_is_loaded_as_original_text_rows(self):
        path = self.directory / "IDS_mapping.csv"
        payload = b'admission_type_id,description\n1,Emergency\n,\nadmission_source_id,description\n7,"Emergency, room"\n'
        path.write_bytes(payload)
        with (
            patch.object(data_loading, "MAPPING_SHA256", sha256(payload).hexdigest()),
            patch.object(data_loading, "MAPPING_ROWS", 4),
        ):
            table = data_loading.load_ids_mapping(path)
        self.assertEqual(table.rows[1], {"admission_type_id": "", "description": ""})
        self.assertEqual(table.rows[2]["admission_type_id"], "admission_source_id")
        self.assertEqual(table.rows[3]["description"], "Emergency, room")
        self.assertEqual(path.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
