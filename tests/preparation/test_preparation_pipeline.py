"""Validate the preparation notebooks from verified raw inputs to saved Weka datasets."""

import asyncio
import os
import sys
from collections import Counter
from hashlib import sha256
from itertools import chain
from pathlib import Path
from shutil import copyfile, copytree, ignore_patterns
from tempfile import TemporaryDirectory
from unittest import SkipTest, TestCase

from src.io_utils import read_csv, read_json, sha256_file, write_json
from src.paths import PROJECT_ROOT
from src.preparation.data_loading import (
    COUNT_COLUMNS, ENCOUNTER_COLUMNS, ENCOUNTER_SHA256, ID_COLUMNS, MAPPING_SHA256,
)
from src.preparation.dataset_export import create_arff_schema, validate_arff
from src.preparation.dataset_preparation import group_icd9


RAW_HASHES = {
    "data/raw/diabetic_data.csv": ENCOUNTER_SHA256,
    "data/raw/IDS_mapping.csv": MAPPING_SHA256,
}
NOTEBOOKS = ("00_data_loading.ipynb", "01_data_preparation.ipynb", "02_split_export.ipynb")
PARTITIONS = ("train", "validation", "test")
MODEL_CSVS = (*PARTITIONS, "train_validation")
ARFF_FILES = ("full_prepared", *MODEL_CSVS, "test_for_train_validation")
OUTPUT_DIRS = ("data/interim", "data/processed", "data/weka", "outputs/preparation")
EXPECTED_OUTPUTS = {
    "data/interim/cleaned.csv", "data/interim/cohort.csv",
    *(f"data/processed/{name}.csv" for name in ("full_prepared", *MODEL_CSVS)),
    *(f"data/weka/{name}.arff" for name in ARFF_FILES),
    *(f"outputs/preparation/{name}" for name in (
        "cleaning_audit.csv", "cohort_audit.csv", "feature_audit.csv",
        "preparation_manifest.json", "split_manifest.csv", "arff_export_audit.json",
    )),
}


async def _execute_notebooks(root: Path) -> None:
    """Run each notebook with a fresh kernel using the test runner's interpreter."""
    import nbformat
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    kernel_directory = root / "kernels"
    write_json(kernel_directory / "validation" / "kernel.json", {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Preparation validation",
        "language": "python",
    })
    specs = KernelSpecManager(kernel_dirs=[str(kernel_directory)], ensure_native_kernel=False)
    env = {
        **os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PYTHONPATH": "",
    }
    for name in ("IPYTHONDIR", "JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR", "JUPYTER_RUNTIME_DIR"):
        directory = root / name
        directory.mkdir()
        env[name] = str(directory)

    for filename in NOTEBOOKS:
        path = root / "notebooks" / filename
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        if any(cell.outputs or cell.execution_count is not None for cell in notebook.cells if cell.cell_type == "code"):
            raise AssertionError(f"{filename} contains saved execution results.")
        manager = AsyncKernelManager(kernel_name="validation", kernel_spec_manager=specs)
        client = NotebookClient(notebook, km=manager, timeout=180)
        ran = await client.async_execute(cwd=str(path.parent), env=env, cleanup_kc=True)
        if any(cell.execution_count is None for cell in ran.cells if cell.cell_type == "code"):
            raise AssertionError(f"{filename} contains a code cell that did not run.")


def _prepare_workspace(root: Path, *, crlf_config: bool = False) -> None:
    """Copy only execution inputs and run the notebook entry points."""
    for directory in ("src", "config", "notebooks"):
        copytree(PROJECT_ROOT / directory, root / directory, ignore=ignore_patterns("__pycache__", ".ipynb_checkpoints"))
    copyfile(PROJECT_ROOT / "requirements.txt", root / "requirements.txt")
    config_path = root / "config/preparation.json"
    config_bytes = config_path.read_bytes().replace(b"\r\n", b"\n")
    config_path.write_bytes(config_bytes.replace(b"\n", b"\r\n") if crlf_config else config_bytes)
    (root / "data/raw").mkdir(parents=True)
    for path in RAW_HASHES:
        copyfile(PROJECT_ROOT / path, root / path)
    asyncio.run(_execute_notebooks(root), loop_factory=asyncio.SelectorEventLoop)


def _output_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for directory in OUTPUT_DIRS
        for path in (root / directory).rglob("*")
        if path.is_file()
    }


class PreparationPipelineTests(TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [path for path in RAW_HASHES if not (PROJECT_ROOT / path).is_file()]
        if missing:
            raise SkipTest("End-to-end preparation requires both UCI files. Missing: " + ", ".join(missing))
        cls.workspace = Path(cls.enterClassContext(TemporaryDirectory(prefix="preparation-validation-")))
        cls.root = cls.workspace / "first"
        _prepare_workspace(cls.root)
        cls.hashes = _output_hashes(cls.root)
        cls.policy = read_json(cls.root / "config/preparation.json")

    def test_raw_integrity_artifact_scope_and_manifests(self):
        self.assertEqual(set(self.hashes), EXPECTED_OUTPUTS)
        for path, expected in RAW_HASHES.items():
            with self.subTest(path=path):
                self.assertEqual(sha256_file(self.root / path), expected)
                self.assertEqual(sha256_file(PROJECT_ROOT / path), expected)
        for name in NOTEBOOKS:
            path = Path("notebooks") / name
            self.assertEqual(sha256_file(self.root / path), sha256_file(PROJECT_ROOT / path))

        manifest = read_json(self.root / "outputs/preparation/preparation_manifest.json")
        self.assertEqual(manifest["raw_sha256"], RAW_HASHES)
        config_bytes = (self.root / "config/preparation.json").read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(manifest["configuration"]["sha256"], sha256(config_bytes).hexdigest())
        self.assertEqual(set(manifest["artifact_sha256"]), {
            "data/interim/cleaned.csv", "data/interim/cohort.csv", "data/processed/full_prepared.csv",
            "outputs/preparation/cleaning_audit.csv", "outputs/preparation/cohort_audit.csv",
            "outputs/preparation/feature_audit.csv",
        })
        for path, checksum in manifest["artifact_sha256"].items():
            self.assertEqual(checksum, self.hashes[path], path)

    def test_cleaning_changes_only_approved_markers(self):
        _, raw = read_csv(self.root / "data/raw/diabetic_data.csv", expected_columns=ENCOUNTER_COLUMNS)
        _, cleaned = read_csv(self.root / "data/interim/cleaned.csv", expected_columns=ENCOUNTER_COLUMNS)
        _, audit = read_csv(self.root / "outputs/preparation/cleaning_audit.csv")
        markers = self.policy["cleaning"]["missing_markers"]
        literals = self.policy["cleaning"]["preserve_literals"]
        changes, literal_counts = Counter(), Counter()
        for number, (source, row) in enumerate(zip(raw, cleaned, strict=True), start=1):
            for column, value in source.items():
                expected = "" if markers.get(column) == value else value
                if row[column] != expected:
                    self.fail(f"Cleaning changed the wrong value in row {number}, column {column}.")
                if expected != value:
                    changes[column] += 1
                if literals.get(column) == value:
                    literal_counts[column] += 1
        self.assertEqual((len(raw), len(cleaned)), (101_766, 101_766))
        self.assertEqual(changes.total(), 192_852)
        self.assertEqual([row["column"] for row in audit], list(ENCOUNTER_COLUMNS))
        for row in audit:
            column = row["column"]
            with self.subTest(column=column):
                self.assertEqual(int(row["changed_cells"]), changes[column])
                self.assertEqual(int(row["empty_after"]), changes[column])
                self.assertEqual(int(row["literal_before"]), literal_counts[column])
                self.assertEqual(int(row["literal_after"]), literal_counts[column])

    def test_cohort_diagnoses_and_predictors_match_the_saved_rows(self):
        _, cleaned = read_csv(self.root / "data/interim/cleaned.csv", expected_columns=ENCOUNTER_COLUMNS)
        excluded = set(self.policy["cohort"]["excluded_discharge_codes"])
        eligible = [row for row in cleaned if row["discharge_disposition_id"] not in excluded]
        eligible.sort(key=lambda row: (int(row["encounter_id"]), row["encounter_id"]))
        representatives = {}
        for row in eligible:
            representatives.setdefault(row["patient_nbr"], row)
        _, cohort = read_csv(self.root / "data/interim/cohort.csv", expected_columns=ENCOUNTER_COLUMNS)
        self.assertEqual((len(eligible), len(cohort)), (99_343, 69_990))

        predictors = self.policy["feature_selection"]["predictors"]
        header = [*ID_COLUMNS, *predictors, "readmitted"]
        _, prepared = read_csv(self.root / "data/processed/full_prepared.csv", expected_columns=header)
        self.assertEqual(len(header), 35)

        manifest = read_json(self.root / "outputs/preparation/preparation_manifest.json")
        self.assertEqual([entry["name"] for entry in manifest["schema"]], header)
        self.assertEqual([entry["role"] for entry in manifest["schema"]], ["identifier"] * 2 + ["predictor"] * 32 + ["target"])
        self.assertEqual(
            [entry["name"] for entry in manifest["schema"] if entry["logical_type"] == "numeric"],
            list(COUNT_COLUMNS),
        )
        groups = {f"diag_{index}_group": f"diag_{index}" for index in (1, 2, 3)}
        diagnoses, empty, labels = Counter(), Counter(), Counter()
        compared = zip(representatives.values(), cohort, prepared, strict=True)
        for number, (expected, selected, projected) in enumerate(compared, start=1):
            if expected != selected:
                self.fail(f"Cohort row {number} differs from the lowest eligible encounter.")
            for column, value in projected.items():
                source = group_icd9(selected[groups[column]]) if column in groups else selected[column]
                if value != source:
                    self.fail(f"Prepared row {number}, column {column} differs from its cohort value.")
                if column in groups:
                    diagnoses[column, value] += 1
                if value == "":
                    empty[column] += 1
            labels[projected["readmitted"]] += 1

        self.assertEqual(labels, {"NO": 41_479, "<30": 6_285, ">30": 22_226})
        self.assertEqual(empty, {"race": 1_919, "gender": 3})
        for index, missing, diabetes in ((1, 10, 5_748), (2, 293, 9_700), (3, 1_224, 12_547)):
            column = f"diag_{index}_group"
            with self.subTest(column=column):
                self.assertEqual(diagnoses[column, "Missing"], missing)
                self.assertEqual(diagnoses[column, "Diabetes"], diabetes)
                self.assertEqual(diagnoses[column, "Unmapped"], 0)
        self.assertEqual(manifest["statistics"]["target_counts"], labels)
        self.assertEqual(manifest["statistics"]["empty_cells_by_column"], {column: empty[column] for column in header})

    def test_partitions_isolate_patients_and_derive_exact_model_csvs(self):
        columns, prepared = read_csv(self.root / "data/processed/full_prepared.csv")
        keys = ("encounter_id", "patient_nbr", "split")
        _, manifest = read_csv(self.root / "outputs/preparation/split_manifest.csv", expected_columns=keys)
        groups = {name: [] for name in PARTITIONS}
        patients = set()
        encounters = set()
        for number, (source, entry) in enumerate(zip(prepared, manifest, strict=True), start=1):
            encounter, patient, name = (entry[key] for key in keys)
            if (encounter, patient) != (source["encounter_id"], source["patient_nbr"]):
                self.fail(f"Split manifest row {number} points to the wrong encounter or patient.")
            if encounter in encounters or patient in patients:
                self.fail(f"Split manifest row {number} repeats an encounter or patient.")
            encounters.add(encounter)
            patients.add(patient)
            groups[name].append(source)
        self.assertEqual(len(patients), 69_990)
        expected_counts = {
            "train": {"NO": 26_547, "<30": 4_022, ">30": 14_225},
            "validation": {"NO": 6_636, "<30": 1_006, ">30": 3_556},
            "test": {"NO": 8_296, "<30": 1_257, ">30": 4_445},
        }
        model_columns = [*columns[2:-1], "class"]
        self.assertEqual(len(model_columns), 33)
        model_tables = {}
        for name, sources in groups.items():
            with self.subTest(split=name):
                _, model = read_csv(self.root / f"data/processed/{name}.csv", expected_columns=model_columns)
                model_tables[name] = model
                self.assertEqual(Counter(row["readmitted"] for row in sources), expected_counts[name])
                for number, (source, projected) in enumerate(zip(sources, model, strict=True), start=1):
                    target = "not_readmitted" if source["readmitted"] == "NO" else "readmitted"
                    if projected["class"] != target or any(
                        projected[column] != source[column] for column in model_columns[:-1]
                    ):
                        self.fail(f"{name}.csv row {number} differs from the manifest-selected cohort row.")

        _, combined = read_csv(self.root / "data/processed/train_validation.csv", expected_columns=model_columns)
        self.assertEqual(len(combined), 55_992)
        ordered = chain(model_tables["train"], model_tables["validation"])
        for number, (expected, actual) in enumerate(zip(ordered, combined, strict=True), start=1):
            if expected != actual:
                self.fail(f"Train + Validation differs at row {number}.")

    def test_arff_headers_and_values_match_csv_and_fitting_scope(self):
        columns, prepared = read_csv(self.root / "data/processed/full_prepared.csv")
        predictors = columns[2:-1]
        _, train = read_csv(self.root / "data/processed/train.csv", expected_columns=[*predictors, "class"])
        _, development = read_csv(
            self.root / "data/processed/train_validation.csv", expected_columns=[*predictors, "class"],
        )
        common = {"target": "class", "target_values": ("not_readmitted", "readmitted")}
        exploration = create_arff_schema(
            predictors, prepared, target="readmitted", target_values=("NO", "<30", ">30"),
        )
        train_schema = create_arff_schema(predictors, train, **common)
        development_schema = create_arff_schema(predictors, development, **common)
        audit = read_json(self.root / "outputs/preparation/arff_export_audit.json")
        self.assertEqual(audit["domains_added_in_final_schema"], {"admission_source_id": ["14"]})
        checks = (
            ("full_prepared", "full_prepared", exploration, False),
            ("train", "train", train_schema, False),
            ("validation", "validation", train_schema, True),
            ("test", "test", train_schema, True),
            ("train_validation", "train_validation", development_schema, False),
            ("test_for_train_validation", "test", development_schema, True),
        )
        for name, source_csv, schema, allow_unseen in checks:
            with self.subTest(export=name):
                rows = prepared if name == "full_prepared" else read_csv(self.root / f"data/processed/{source_csv}.csv")[1]
                path = Path(f"data/weka/{name}.arff")
                content = (self.root / path).read_bytes().decode("utf-8")
                relation = "diabetes_readmission_full_prepared" if name == "full_prepared" else "diabetes_readmission_32_features"
                self.assertEqual(len(schema), 33)
                self.assertEqual([column.name for column in schema if column.domain is None], list(COUNT_COLUMNS))
                validate_arff(content, relation, schema, rows, allow_unseen=allow_unseen)
                result = audit["datasets"][name]
                self.assertEqual((result["rows"], result["attributes"]), (len(rows), len(schema)))
                self.assertEqual(result["sha256"], self.hashes[path.as_posix()])
        self.assertEqual(audit["datasets"]["validation"]["unseen_nominal_values_by_column"], {
            "admission_source_id": {"14": 2},
        })
        for name in ("test", "test_for_train_validation"):
            self.assertEqual(audit["datasets"][name]["unseen_nominal_values_by_column"], {
                "admission_source_id": {"11": 1}, "chlorpropamide": {"Down": 1},
            })

    def test_second_clean_notebook_run_reproduces_all_outputs(self):
        replay = self.workspace / "replay"
        _prepare_workspace(replay, crlf_config=True)
        self.assertEqual(_output_hashes(replay), self.hashes)
        for path, expected in RAW_HASHES.items():
            self.assertEqual(sha256_file(replay / path), expected)
