"""Check the approved predictor schema, projection, and audit."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.io_utils import read_csv, read_json, write_json
from src.preparation.data_loading import ENCOUNTER_COLUMNS
from src.preparation.dataset_preparation import (
    FEATURE_AUDIT_COLUMNS,
    PREPARATION_CONFIG_PATH,
    group_diagnoses,
    save_feature_audit,
    select_predictors,
)


EXPECTED_PREDICTORS = tuple("""
race
gender
age
admission_type_id
admission_source_id
time_in_hospital
num_lab_procedures
num_procedures
num_medications
number_outpatient
number_emergency
number_inpatient
number_diagnoses
max_glu_serum
A1Cresult
metformin
repaglinide
nateglinide
chlorpropamide
glimepiride
glipizide
glyburide
pioglitazone
rosiglitazone
acarbose
insulin
glyburide-metformin
change
diabetesMed
diag_1_group
diag_2_group
diag_3_group
""".split())


class FeatureSelectionTests(TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(TemporaryDirectory()))
        row = dict.fromkeys(ENCOUNTER_COLUMNS, "0")
        row.update(
            encounter_id="001", patient_nbr="007", readmitted="<30",
            weight="", race="", A1Cresult="None", diag_1="250.01", diag_2="", diag_3="401",
        )
        self.grouped = group_diagnoses(ENCOUNTER_COLUMNS, [row])

    def test_applies_approved_order_and_preserves_selected_values(self):
        before = deepcopy(self.grouped.rows)
        result = select_predictors(self.grouped)
        expected_columns = ("encounter_id", "patient_nbr", *EXPECTED_PREDICTORS, "readmitted")
        self.assertEqual(result.columns, expected_columns)
        self.assertEqual(tuple(result.rows[0]), expected_columns)
        self.assertEqual(result.rows[0], {column: before[0][column] for column in expected_columns})
        self.assertEqual(self.grouped.rows, before)
        self.assertEqual(result.rows[0]["diag_2_group"], "Missing")
        self.assertEqual(result.rows[0]["A1Cresult"], "None")
        self.assertEqual(result.rows[0]["readmitted"], "<30")

    def test_rejects_invalid_feature_configurations(self):
        path = self.directory / "preparation.json"
        cases = (
            ("encounter_id", "unique column names"),
            ("unrecognized", "account for the grouped schema"),
        )
        for column, error in cases:
            with self.subTest(column=column):
                config = read_json(PREPARATION_CONFIG_PATH)
                config["feature_selection"]["predictors"][0] = column
                write_json(path, config, overwrite=True)
                with self.assertRaisesRegex(ValueError, error):
                    select_predictors(self.grouped, config_path=path)

        config = read_json(PREPARATION_CONFIG_PATH)
        del config["feature_selection"]["dropped_source_columns"]["weight"]
        write_json(path, config, overwrite=True)
        with self.assertRaisesRegex(ValueError, "account for the grouped schema"):
            select_predictors(self.grouped, config_path=path)

    def test_audit_distinguishes_removed_replaced_and_uncreated_columns(self):
        result = select_predictors(self.grouped)
        audit = {entry["column"]: entry for entry in result.audit}
        self.assertEqual(len(audit), 60)
        self.assertEqual(audit["weight"]["action"], "drop_source")
        self.assertEqual(audit["weight"]["empty_before"], 1)
        self.assertIsNone(audit["weight"]["empty_after"])
        self.assertEqual(audit["diag_2"]["action"], "replaced_diagnosis")
        self.assertEqual(audit["diag_2"]["output_column"], "diag_2_group")
        self.assertEqual(audit["diag_2"]["empty_before"], 1)
        self.assertEqual(audit["diag_2_group"]["empty_after"], 0)
        self.assertEqual(audit["total_prior_visits"]["action"], "not_created")
        self.assertIsNone(audit["total_prior_visits"]["rows_before"])
        self.assertEqual(audit["readmitted"]["output_position"], 35)
        self.assertEqual(audit["race"]["empty_after"], 1)
        self.assertEqual(sum(entry["logical_type"] == "numeric" for entry in audit.values()), 8)

        path = self.directory / "feature_audit.csv"
        save_feature_audit(result, audit_path=path)
        save_feature_audit(result, audit_path=path)
        columns, rows = read_csv(path)
        self.assertEqual(columns, list(FEATURE_AUDIT_COLUMNS))
        self.assertEqual(len(rows), len(audit))
