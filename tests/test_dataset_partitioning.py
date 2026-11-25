"""Verify deterministic stratification and partition CSV exports."""

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.io_utils import read_csv, read_json
from src.preparation.dataset_export import (
    SPLIT_CONFIG_PATH,
    TARGET_MAPPING_CONFIG_PATH,
    make_split_assignments,
    map_binary_targets,
    save_model_partitions,
)
from src.preparation.dataset_preparation import PREPARATION_CONFIG_PATH


class PartitionTests(TestCase):
    def setUp(self):
        predictors = read_json(PREPARATION_CONFIG_PATH)["feature_selection"]["predictors"]
        self.columns = ["encounter_id", "patient_nbr", *predictors, "readmitted"]
        self.rows = []
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
            self.rows.append(row)

    def test_rank_order_and_largest_remainder_class_tie(self):
        assignments = make_split_assignments(self.rows)
        self.assertEqual({key for key, name in assignments.items() if name == "test"}, {
            "1", "10", "12", "14", "24", "29",
        })
        self.assertEqual({key for key, name in assignments.items() if name == "validation"}, {
            "6", "9", "13", "15", "30",
        })
        self.assertEqual(set(assignments), {row["encounter_id"] for row in self.rows})
        self.assertEqual(assignments, make_split_assignments(self.rows[::-1]))

    def test_duplicate_patients_and_encounters_are_rejected(self):
        for key in ("patient_nbr", "encounter_id"):
            with self.subTest(key=key):
                invalid = [self.rows[0], self.rows[1] | {key: self.rows[0][key]}, *self.rows[2:]]
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
            mapped = map_binary_targets(self.rows)
            summary = save_model_partitions(self.columns, mapped[::-1], project_root=root)

            columns, train = read_csv(root / "data/processed/train.csv")
            _, validation = read_csv(root / "data/processed/validation.csv", expected_columns=columns)
            _, test = read_csv(root / "data/processed/test.csv", expected_columns=columns)
            _, combined = read_csv(root / "data/processed/train_validation.csv", expected_columns=columns)
            manifest_columns, manifest = read_csv(root / "outputs/preparation/split_manifest.csv")

            self.assertEqual(columns, self.columns[2:-1] + ["class"])
            self.assertEqual((len(train), len(validation), len(test), len(combined)), (19, 5, 6, 24))
            self.assertEqual(combined, train + validation)
            self.assertEqual(manifest_columns, ["encounter_id", "patient_nbr", "split"])
            self.assertEqual([row["encounter_id"] for row in manifest], [str(i) for i in range(1, 31)])
            self.assertEqual(len({row["patient_nbr"] for row in manifest}), 30)
            for name, rows in (("train", train), ("validation", validation), ("test", test)):
                encounter_ids = [row["encounter_id"] for row in manifest if row["split"] == name]
                self.assertEqual([row["num_medications"] for row in rows], encounter_ids)
            self.assertTrue(all(row["A1Cresult"] == "None" for row in combined))
            self.assertEqual(summary["validation"]["original_labels"], {"NO": 2, "<30": 2, ">30": 1})
            self.assertEqual(summary["validation"]["binary_classes"], {"not_readmitted": 2, "readmitted": 3})
            self.assertEqual(summary["validation"]["empty_cells_by_column"]["race"], 2)
            self.assertEqual(summary, save_model_partitions(self.columns, mapped, project_root=root))

    def test_existing_mismatched_export_is_preserved_without_creating_other_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_export_configuration(root)
            test_path = root / "data/processed/test.csv"
            test_path.parent.mkdir(parents=True)
            test_path.write_bytes(b"stale\n")
            with self.assertRaisesRegex(FileExistsError, "Existing output differs"):
                save_model_partitions(self.columns, map_binary_targets(self.rows), project_root=root)
            self.assertEqual(test_path.read_bytes(), b"stale\n")
            self.assertEqual(list(test_path.parent.iterdir()), [test_path])
            self.assertFalse((root / "outputs").exists())

    def test_inconsistent_binary_class_is_detected_before_export(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_export_configuration(root)
            mapped = map_binary_targets(self.rows)
            mapped[0]["class"] = "readmitted"
            with self.assertRaisesRegex(ValueError, "Binary class disagrees"):
                save_model_partitions(self.columns, mapped, project_root=root)
            self.assertFalse((root / "data").exists())
