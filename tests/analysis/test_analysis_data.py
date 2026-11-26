"""Verify the Analysis reader against a prepared fixture."""

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from src.analysis.data import ANALYSIS_CONFIG_PATH, load_analysis_inputs
from src.io_utils import read_json, sha256_file, write_csv, write_json
from src.paths import PROJECT_ROOT
from src.preparation.data_loading import ID_COLUMNS, TARGET_COLUMN
from src.preparation.dataset_export import TARGET_MAPPING_CONFIG_PATH
from src.preparation.dataset_preparation import (
    FULL_PREPARED_PATH,
    PREPARATION_CONFIG_PATH,
    PREPARATION_MANIFEST_PATH,
)


class AnalysisInputTests(TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        (self.root / "config").mkdir()
        for source in (ANALYSIS_CONFIG_PATH, PREPARATION_CONFIG_PATH, TARGET_MAPPING_CONFIG_PATH):
            copyfile(source, self.root / source.relative_to(PROJECT_ROOT))

        self.predictor_columns = read_json(PREPARATION_CONFIG_PATH)["feature_selection"]["predictors"]
        self.columns = [*ID_COLUMNS, *self.predictor_columns, TARGET_COLUMN]
        self.rows = []
        for index, label in enumerate(("NO", "<30", ">30"), start=1):
            row = dict.fromkeys(self.columns, "0")
            row.update(
                encounter_id=str(index), patient_nbr=str(100 + index), readmitted=label,
                A1Cresult="None", max_glu_serum="None",
            )
            self.rows.append(row)
        self.rows[0]["max_glu_serum"] = ""
        self.rows[1]["A1Cresult"] = ">7"
        self.rows[1]["number_inpatient"] = ""
        self.rows[2]["number_inpatient"] = "2"

    def save_fixture(self, *, overwrite=False):
        data_key = FULL_PREPARED_PATH.relative_to(PROJECT_ROOT).as_posix()
        config_key = PREPARATION_CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix()
        frame = pd.DataFrame(self.rows, columns=self.columns, dtype="string")
        csv_path = write_csv(self.root / data_key, frame, overwrite=overwrite)
        write_json(self.root / PREPARATION_MANIFEST_PATH.relative_to(PROJECT_ROOT), {
            "dataset_path": data_key,
            "configuration": {"path": config_key, "sha256": sha256_file(self.root / config_key)},
            "artifact_sha256": {data_key: sha256_file(csv_path)},
            "schema": [{"name": column} for column in self.columns],
            "statistics": {"rows": 3, "columns": 35, "target_counts": {"NO": 1, "<30": 1, ">30": 1}},
        }, overwrite=overwrite)

    def test_aligned_binary_target_and_literal_lab_values(self):
        self.save_fixture()
        analysis = load_analysis_inputs(project_root=self.root)
        self.assertEqual(list(analysis.predictors), self.predictor_columns)
        self.assertTrue(analysis.target.index.equals(analysis.predictors.index))
        self.assertEqual(analysis.target.tolist(), ["not_readmitted", "readmitted", "readmitted"])
        self.assertEqual(analysis.predictors["A1Cresult"].tolist(), ["None", ">7", "None"])
        self.assertEqual(analysis.predictors["max_glu_serum"].tolist(), ["", "None", "None"])
        self.assertEqual(str(analysis.predictors["number_inpatient"].dtype), "Int64")
        self.assertEqual(analysis.predictors["number_inpatient"].fillna(-1).tolist(), [0, -1, 2])
        self.assertTrue(pd.isna(analysis.predictors.loc[1, "number_inpatient"]))

    def test_duplicate_encounter_or_patient_is_rejected(self):
        for key in ID_COLUMNS:
            with self.subTest(key=key):
                original = self.rows[1][key]
                self.rows[1][key] = self.rows[0][key]
                self.save_fixture(overwrite=True)
                self.rows[1][key] = original
                with self.assertRaisesRegex(ValueError, f"Duplicate prepared {key}"):
                    load_analysis_inputs(project_root=self.root)

    def test_invalid_original_label_is_rejected(self):
        self.rows[1][TARGET_COLUMN] = "INVALID"
        self.save_fixture()
        with self.assertRaisesRegex(ValueError, "Unexpected target value"):
            load_analysis_inputs(project_root=self.root)

    def test_invalid_count_is_rejected(self):
        self.rows[2]["number_inpatient"] = "-1"
        self.save_fixture()
        with self.assertRaisesRegex(ValueError, "number_inpatient.*nonnegative integers"):
            load_analysis_inputs(project_root=self.root)

    def test_schema_mismatch_is_rejected(self):
        self.columns[2], self.columns[3] = self.columns[3], self.columns[2]
        self.save_fixture()
        with self.assertRaisesRegex(ValueError, "Unexpected columns or column order"):
            load_analysis_inputs(project_root=self.root)
