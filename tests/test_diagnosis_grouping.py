"""Check fixed ICD-9 rules and diagnosis replacement."""

from copy import deepcopy
from unittest import TestCase

from src.preparation.data_loading import ENCOUNTER_COLUMNS
from src.preparation.dataset_preparation import DIAGNOSIS_COLUMNS, group_diagnoses, group_icd9


class DiagnosisGroupingTests(TestCase):
    def test_numeric_ranges_and_special_codes(self):
        cases = {
            "Neoplasms": ("140", "239.99"),
            "Diabetes": ("250", "250.0", "250.99", " 250.13 "),
            "Circulatory": ("390", "459.99", "785.2"),
            "Respiratory": ("460", "519.99", "786.1"),
            "Digestive": ("520", "579.99", "787"),
            "Genitourinary": ("580", "629.99", "788.01"),
            "Musculoskeletal": ("710", "739.99"),
            "Injury": ("800", "999.99"),
            "Other_numeric": ("001", "139.99", "240", "249.99", "251", "389.99", "630",
                              "709.99", "740", "784.99", "789", "799.99"),
        }
        for expected, codes in cases.items():
            for code in codes:
                with self.subTest(code=code):
                    self.assertEqual(group_icd9(code), expected)

    def test_supplementary_missing_and_unmapped_codes(self):
        cases = {
            "Other_V": ("V10", " v10.01 ", "V1234.99"),
            "Other_E": ("E800", " e999.9 ", "E1"),
            "Missing": ("",),
            "Unmapped": ("?", "   ", "V", "E12345", "250.", "250.123", "2e2", "1000", "-250", "250x", "\u0662\u0665\u0660"),
        }
        for expected, codes in cases.items():
            for code in codes:
                with self.subTest(code=code):
                    self.assertEqual(group_icd9(code), expected)

    def test_replaces_exactly_three_columns_and_audits_each_position(self):
        row = dict.fromkeys(ENCOUNTER_COLUMNS, "0")
        row.update(encounter_id="001", patient_nbr="007", readmitted="<30", A1Cresult="None")
        rows = [
            row | {"diag_1": "250.01", "diag_2": "", "diag_3": "E800"},
            row | {"diag_1": "invalid", "diag_2": "V12", "diag_3": "250"},
        ]
        before = deepcopy(rows)
        result = group_diagnoses(ENCOUNTER_COLUMNS, rows)
        self.assertEqual(rows, before)
        expected_columns = tuple(
            f"{column}_group" if column in DIAGNOSIS_COLUMNS else column for column in ENCOUNTER_COLUMNS
        )
        self.assertEqual(result.columns, expected_columns)
        for source, grouped in zip(before, result.rows, strict=True):
            self.assertEqual(tuple(grouped), result.columns)
            for column in ENCOUNTER_COLUMNS:
                if column not in DIAGNOSIS_COLUMNS:
                    self.assertEqual(grouped[column], source[column])
        self.assertEqual(result.group_counts["diag_1_group"]["Diabetes"], 1)
        self.assertEqual(result.group_counts["diag_1_group"]["Unmapped"], 1)
        self.assertEqual(result.group_counts["diag_2_group"]["Missing"], 1)
        self.assertEqual(result.group_counts["diag_2_group"]["Other_V"], 1)
        self.assertEqual(result.group_counts["diag_3_group"]["Other_E"], 1)
        self.assertTrue(all(sum(counts.values()) == 2 for counts in result.group_counts.values()))
