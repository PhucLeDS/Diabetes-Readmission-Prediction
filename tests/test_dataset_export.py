"""Verify the Any Readmission target and the inputs to future dataset exports."""

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.io_utils import read_json, sha256_file, write_csv, write_json
from src.paths import PROJECT_ROOT
from src.preparation.dataset_export import (
    TARGET_MAPPING_CONFIG_PATH,
    build_model_dataset,
    load_full_prepared,
    load_target_mapping,
    map_binary_targets,
)
from src.preparation.dataset_preparation import (
    FULL_PREPARED_PATH,
    PREPARATION_CONFIG_PATH,
    PREPARATION_MANIFEST_PATH,
)


class BinaryTargetTests(TestCase):
    def setUp(self):
        predictors = read_json(PREPARATION_CONFIG_PATH)["feature_selection"]["predictors"]
        self.columns = ["encounter_id", "patient_nbr", *predictors, "readmitted"]
        self.rows = []
        for index, label in enumerate(("NO", "<30", ">30"), start=1):
            row = dict.fromkeys(self.columns, "0")
            row.update(encounter_id=str(index), patient_nbr=str(100 + index), readmitted=label)
            self.rows.append(row)
        self.rows[0]["race"] = ""
        self.rows[1]["A1Cresult"] = "None"

    def test_all_source_labels_and_model_projection(self):
        original = [row.copy() for row in self.rows]
        mapped = map_binary_targets(self.rows)
        columns, model = build_model_dataset(self.columns, mapped)
        expected_classes = ["not_readmitted", "readmitted", "readmitted"]
        self.assertEqual([row["class"] for row in mapped], expected_classes)
        self.assertEqual([row["class"] for row in model], expected_classes)
        self.assertEqual(self.rows, original)
        self.assertEqual(columns, [*self.columns[2:-1], "class"])
        self.assertEqual(len(columns), 33)
        for source, enriched, predictors in zip(original, mapped, model, strict=True):
            self.assertEqual(list(enriched), [*self.columns, "class"])
            self.assertEqual({name: enriched[name] for name in self.columns}, source)
            self.assertEqual(list(predictors), columns)
            self.assertEqual({name: predictors[name] for name in columns[:-1]}, {
                name: source[name] for name in columns[:-1]
            })

    def test_invalid_source_labels_are_rejected(self):
        for value in ("", "INVALID"):
            with self.subTest(value=value):
                bad = self.rows[1] | {"readmitted": value}
                with self.assertRaisesRegex(ValueError, "Unexpected target value in data row 2"):
                    map_binary_targets([self.rows[0], bad])

    def test_mapping_does_not_replace_an_existing_class(self):
        with self.assertRaisesRegex(ValueError, "already contain"):
            map_binary_targets([self.rows[0] | {"class": "readmitted"}])

    def test_early_readmission_mapping_is_rejected(self):
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "target_mapping.json"
            config = read_json(TARGET_MAPPING_CONFIG_PATH)
            config["class_mapping"][">30"] = "not_readmitted"
            write_json(config_path, config)
            with self.assertRaisesRegex(ValueError, "Any Readmission"):
                load_target_mapping(config_path)

    def test_projection_rejects_wrong_schema_and_class(self):
        mapped = map_binary_targets(self.rows)
        with self.assertRaisesRegex(ValueError, "32 predictors"):
            build_model_dataset(self.columns[1:], mapped)
        mapped[0]["class"] = "NO"
        with self.assertRaisesRegex(ValueError, "Unexpected target value"):
            build_model_dataset(self.columns, mapped)

    def save_fixture(self, root):
        data_key = FULL_PREPARED_PATH.relative_to(PROJECT_ROOT)
        config_key = PREPARATION_CONFIG_PATH.relative_to(PROJECT_ROOT)
        manifest_key = PREPARATION_MANIFEST_PATH.relative_to(PROJECT_ROOT)
        (root / config_key).parent.mkdir(parents=True)
        copyfile(PREPARATION_CONFIG_PATH, root / config_key)
        csv_path = write_csv(root / data_key, self.columns, self.rows)
        manifest_path = write_json(root / manifest_key, {
            "dataset_path": data_key.as_posix(),
            "configuration": {"path": config_key.as_posix(), "sha256": sha256_file(root / config_key)},
            "artifact_sha256": {data_key.as_posix(): sha256_file(csv_path)},
            "schema": [{"name": column} for column in self.columns],
            "statistics": {"rows": 3, "columns": 35, "target_counts": {"NO": 1, "<30": 1, ">30": 1}},
        })
        return csv_path, manifest_path

    def test_loader_reads_verified_values_and_rejects_changed_csv(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path, _ = self.save_fixture(root)
            self.assertEqual(load_full_prepared(project_root=root), (self.columns, self.rows))
            with csv_path.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "CSV differs from the preparation manifest"):
                load_full_prepared(project_root=root)

    def test_loader_rejects_manifest_schema_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest_path = self.save_fixture(root)
            manifest = read_json(manifest_path)
            manifest["schema"][0]["name"] = "wrong"
            write_json(manifest_path, manifest, overwrite=True)
            with self.assertRaisesRegex(ValueError, "Unexpected columns or column order"):
                load_full_prepared(project_root=root)

    def test_loader_allows_checkout_line_endings_but_rejects_config_edits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.save_fixture(root)
            config_path = root / PREPARATION_CONFIG_PATH.relative_to(PROJECT_ROOT)
            config_lf = config_path.read_bytes().replace(b"\r\n", b"\n")
            config_path.write_bytes(
                config_lf if b"\r\n" in config_path.read_bytes() else config_lf.replace(b"\n", b"\r\n")
            )
            self.assertEqual(load_full_prepared(project_root=root), (self.columns, self.rows))
            config = read_json(config_path)
            config["feature_selection"]["predictors"][0] = "new_predictor"
            write_json(config_path, config, overwrite=True)
            with self.assertRaisesRegex(ValueError, "configuration differs from the manifest"):
                load_full_prepared(project_root=root)
