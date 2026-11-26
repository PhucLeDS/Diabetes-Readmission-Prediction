"""Check the approved predictor schema, projection, and audit."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

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
        self.grouped = group_diagnoses(pd.DataFrame([row], dtype="string"))

    def test_applies_approved_order_and_preserves_selected_values(self):
        before = self.grouped.frame.copy()
        result = select_predictors(self.grouped)
        expected_columns = ["encounter_id", "patient_nbr", *EXPECTED_PREDICTORS, "readmitted"]
        pd.testing.assert_frame_equal(result.frame, before.loc[:, expected_columns])
        pd.testing.assert_frame_equal(self.grouped.frame, before)
        self.assertEqual(result.frame.at[0, "diag_2_group"], "Missing")
        self.assertEqual(result.frame.at[0, "A1Cresult"], "None")
        self.assertEqual(result.frame.at[0, "readmitted"], "<30")

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
        audit = result.audit.set_index("column")
        self.assertEqual(len(audit), 60)
        self.assertEqual(audit.at["weight", "action"], "drop_source")
        self.assertEqual(audit.at["weight", "empty_before"], 1)
        self.assertTrue(pd.isna(audit.at["weight", "empty_after"]))
        self.assertEqual(audit.at["diag_2", "action"], "replaced_diagnosis")
        self.assertEqual(audit.at["diag_2", "output_column"], "diag_2_group")
        self.assertEqual(audit.at["diag_2", "empty_before"], 1)
        self.assertEqual(audit.at["diag_2_group", "empty_after"], 0)
        self.assertEqual(audit.at["total_prior_visits", "action"], "not_created")
        self.assertTrue(pd.isna(audit.at["total_prior_visits", "rows_before"]))
        self.assertEqual(audit.at["readmitted", "output_position"], 35)
        self.assertEqual(audit.at["race", "empty_after"], 1)
        self.assertEqual(audit["logical_type"].eq("numeric").sum(), 8)

        path = self.directory / "feature_audit.csv"
        save_feature_audit(result, audit_path=path)
        save_feature_audit(result, audit_path=path)
        columns, rows = read_csv(path)
        self.assertEqual(columns, list(FEATURE_AUDIT_COLUMNS))
        self.assertEqual(len(rows), len(audit))
        by_name = {row["column"]: row for row in rows}
        self.assertEqual(by_name["readmitted"]["output_position"], "35")
        self.assertEqual(by_name["weight"]["empty_after"], "")
