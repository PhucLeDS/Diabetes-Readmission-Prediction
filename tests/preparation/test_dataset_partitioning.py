"""Verify deterministic stratification and partition CSV exports."""

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from src.io_utils import read_csv, read_json
from src.preparation.dataset_export import (
    SPLIT_CONFIG_PATH,
    TARGET_MAPPING_CONFIG_PATH,
    load_target_mapping,
    make_split_assignments,
    map_binary_targets,
    save_model_partitions,
)
from src.preparation.dataset_preparation import PREPARATION_CONFIG_PATH


class PartitionTests(TestCase):
    def setUp(self):
        predictors = read_json(PREPARATION_CONFIG_PATH)["feature_selection"]["predictors"]
        self.columns = ["encounter_id", "patient_nbr", *predictors, "readmitted"]
        rows = []
        for index in range(1, 31):
            row = dict.fromkeys(self.columns, "0")
            row.update(
                encounter_id=str(index),
                patient_nbr=str(index + 100),
                readmitted=("NO", "<30", ">30")[(index - 1) // 10],
                race="" if index % 2 == 0 else "Other",
                A1Cresult="None",
                num_medications=str(index),
            )
            rows.append(row)
        self.frame = pd.DataFrame(rows, columns=self.columns, dtype="string")
        self.mapping = load_target_mapping()

    def test_rank_order_and_largest_remainder_class_tie(self):
        assignments = make_split_assignments(self.frame)
        self.assertEqual({key for key, name in assignments.items() if name == "test"}, {
            "1", "10", "12", "14", "24", "29",
        })
        self.assertEqual({key for key, name in assignments.items() if name == "validation"}, {
            "6", "9", "13", "15", "30",
        })
        self.assertEqual(set(assignments), set(self.frame["encounter_id"]))
        self.assertEqual(assignments, make_split_assignments(self.frame.iloc[::-1]))

    def test_duplicate_patients_and_encounters_are_rejected(self):
        for key in ("patient_nbr", "encounter_id"):
            with self.subTest(key=key):
                invalid = self.frame.copy()
                invalid.at[1, key] = invalid.at[0, key]
                with self.assertRaisesRegex(ValueError, f"Duplicate {key}"):
                    make_split_assignments(invalid)

    def copy_export_configuration(self, root):
        (root / "config").mkdir()
        for path in (SPLIT_CONFIG_PATH, TARGET_MAPPING_CONFIG_PATH):
            copyfile(path, root / "config" / path.name)

    def test_model_exports_round_trip_and_concatenate_in_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_export_configuration(root)
            mapped = map_binary_targets(self.frame, mapping=self.mapping)
            summary = save_model_partitions(mapped.iloc[::-1], project_root=root)

            columns = self.columns[2:-1] + ["class"]
            train, validation, test, combined = (
                read_csv(root / f"data/processed/{name}.csv", expected_columns=columns)
                for name in ("train", "validation", "test", "train_validation")
            )
            manifest = read_csv(
                root / "outputs/preparation/split_manifest.csv",
                expected_columns=["encounter_id", "patient_nbr", "split"],
            )

            self.assertEqual((len(train), len(validation), len(test), len(combined)), (19, 5, 6, 24))
            pd.testing.assert_frame_equal(combined, pd.concat([train, validation], ignore_index=True))
            self.assertEqual(manifest["encounter_id"].tolist(), [str(i) for i in range(1, 31)])
            self.assertEqual(manifest["patient_nbr"].nunique(), 30)
            for name, partition in (("train", train), ("validation", validation), ("test", test)):
                encounter_ids = manifest.loc[manifest["split"].eq(name), "encounter_id"].tolist()
                self.assertEqual(partition["num_medications"].tolist(), encounter_ids)
            self.assertTrue(combined["A1Cresult"].eq("None").all())
            self.assertEqual(summary["validation"]["original_labels"], {"NO": 2, "<30": 2, ">30": 1})
            self.assertEqual(summary["validation"]["binary_classes"], {"not_readmitted": 2, "readmitted": 3})
            self.assertEqual(summary["validation"]["empty_cells_by_column"]["race"], 2)
            self.assertEqual(summary, save_model_partitions(mapped, project_root=root))

    def test_existing_mismatched_export_is_preserved_without_creating_other_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_export_configuration(root)
            test_path = root / "data/processed/test.csv"
            test_path.parent.mkdir(parents=True)
            test_path.write_bytes(b"stale\n")
            with self.assertRaisesRegex(FileExistsError, "Existing output differs"):
                save_model_partitions(map_binary_targets(self.frame, mapping=self.mapping), project_root=root)
            self.assertEqual(test_path.read_bytes(), b"stale\n")
            self.assertEqual(list(test_path.parent.iterdir()), [test_path])
            self.assertFalse((root / "outputs").exists())

    def test_inconsistent_binary_class_is_detected_before_export(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_export_configuration(root)
            mapped = map_binary_targets(self.frame, mapping=self.mapping)
            mapped.at[0, "class"] = "readmitted"
            with self.assertRaisesRegex(ValueError, "Binary class disagrees"):
                save_model_partitions(mapped, project_root=root)
            self.assertFalse((root / "data").exists())
