"""Check the full prepared dataset contract before CSV export."""

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from src.io_utils import read_json, write_json

from src.preparation.data_loading import ENCOUNTER_COLUMNS
from src.preparation.dataset_preparation import (
    PREPARATION_CONFIG_PATH,
    CohortResult,
    group_diagnoses,
    save_full_prepared,
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
            frame=pd.DataFrame([first, second], dtype="string"),
            flow=[], excluded_by_code={}, unclear_retained_by_code={}, audit=pd.DataFrame(),
        )
        self.selection = select_predictors(group_diagnoses(self.cohort.frame))

    def test_validation_reports_original_labels_and_empty_fields(self):
        statistics = validate_full_prepared(self.selection, self.cohort)
        self.assertEqual((statistics["rows"], statistics["columns"]), (2, 35))
        self.assertEqual(statistics["target_counts"], {"NO": 1, "<30": 1, ">30": 0})
        self.assertEqual(statistics["empty_cells_by_column"]["race"], 1)
        self.assertEqual(statistics["empty_cells_by_column"]["diag_2_group"], 0)
        self.assertEqual(statistics["rows_with_empty_fields"], 1)

    def test_validation_rejects_changed_values_and_grouping(self):
        frame = self.selection.frame
        for column, value in (("readmitted", "NO"), ("diag_2_group", "Unmapped"), ("A1Cresult", "")):
            with self.subTest(column=column):
                original = frame.at[0, column]
                frame.at[0, column] = value
                with self.assertRaisesRegex(ValueError, f"differs from the cohort.*{column}"):
                    validate_full_prepared(self.selection, self.cohort)
                frame.at[0, column] = original

    def test_validation_rejects_reordered_rows_and_duplicate_patients(self):
        original = self.selection.frame
        self.selection.frame = original.iloc[::-1]
        with self.assertRaisesRegex(ValueError, "differs from the cohort.*encounter_id"):
            validate_full_prepared(self.selection, self.cohort)
        self.selection.frame = original
        self.selection.frame.at[1, "patient_nbr"] = "101"
        with self.assertRaisesRegex(ValueError, "Duplicate prepared patient_nbr"):
            validate_full_prepared(self.selection, self.cohort)


    def test_export_reuses_equivalent_manifest_and_rejects_changes(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        for relative in (
            "data/raw/diabetic_data.csv",
            "data/raw/IDS_mapping.csv",
            "data/interim/cleaned.csv",
            "data/interim/cohort.csv",
            "outputs/preparation/cleaning_audit.csv",
            "outputs/preparation/cohort_audit.csv",
            "outputs/preparation/feature_audit.csv",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")

        config_path = root / "config/preparation.json"
        config_path.parent.mkdir()
        lf_config = PREPARATION_CONFIG_PATH.read_bytes().replace(b"\r\n", b"\n")
        crlf_config = lf_config.replace(b"\n", b"\r\n")
        config_path.write_bytes(crlf_config)
        fixture_checksum = sha256(b"fixture").hexdigest()
        self.enterContext(patch.multiple(
            "src.preparation.dataset_preparation",
            ENCOUNTER_SHA256=fixture_checksum,
            MAPPING_SHA256=fixture_checksum,
        ))

        _, manifest_path, manifest = save_full_prepared(self.selection, self.cohort, project_root=root)
        self.assertEqual(manifest["configuration"]["sha256"], sha256(lf_config).hexdigest())

        configurations = (
            (sha256(lf_config).hexdigest(), crlf_config),
            (sha256(crlf_config).hexdigest(), lf_config),
        )
        for earlier_checksum, checkout in configurations:
            with self.subTest(earlier_checksum=earlier_checksum):
                manifest["configuration"]["sha256"] = earlier_checksum
                write_json(manifest_path, manifest, overwrite=True)
                config_path.write_bytes(checkout)
                earlier_bytes = manifest_path.read_bytes()
                _, _, saved = save_full_prepared(self.selection, self.cohort, project_root=root)
                self.assertEqual(saved, manifest)
                self.assertEqual(manifest_path.read_bytes(), earlier_bytes)

        manifest["statistics"]["rows"] += 1
        write_json(manifest_path, manifest, overwrite=True)
        earlier_bytes = manifest_path.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "preparation_manifest"):
            save_full_prepared(self.selection, self.cohort, project_root=root)
        self.assertEqual(manifest_path.read_bytes(), earlier_bytes)

        manifest["statistics"]["rows"] -= 1
        write_json(manifest_path, manifest, overwrite=True)
        earlier_bytes = manifest_path.read_bytes()
        config = read_json(config_path)
        config["cohort"]["excluded_discharge_codes"].remove("21")
        write_json(config_path, config, overwrite=True)
        with self.assertRaisesRegex(FileExistsError, "preparation_manifest"):
            save_full_prepared(self.selection, self.cohort, project_root=root)
        self.assertEqual(manifest_path.read_bytes(), earlier_bytes)
