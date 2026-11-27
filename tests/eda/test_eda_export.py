"""Check the aggregate EDA bundle."""

from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import SkipTest, TestCase
from unittest.mock import patch

import pandas as pd

from src.eda.data import _admission_descriptions, _load_source_cohort
from src.eda.export import INPUT_PATHS, MANIFEST_PATH, TABLE_DIR, export_eda
from src.io_utils import read_csv, read_json, sha256_file, write_csv
from src.paths import PROJECT_ROOT
from src.preparation.data_loading import ENCOUNTER_COLUMNS, MAPPING_COLUMNS, RawTable
from src.preparation.dataset_preparation import COHORT_DATA_PATH


class AdmissionLookupTests(TestCase):
    def test_distinguishes_unmapped_codes_from_unclear_descriptions(self) -> None:
        entries = (
            ("6", "NULL"), ("", ""), ("admission_source_id", "description"), ("7", "Emergency Room"),
        )
        mapping = pd.DataFrame(entries, columns=MAPPING_COLUMNS, dtype="string")
        table = RawTable(Path("mapping.csv"), "", 0, mapping)
        cohort = pd.DataFrame({"admission_type_id": ["6", "99"], "admission_source_id": ["7", ""]})
        with patch("src.eda.data.load_ids_mapping", return_value=table):
            lookup = _admission_descriptions(cohort, table.path).set_index(["column", "code"])
        self.assertEqual(lookup.loc[("admission_type_id", "6"), "description"], "NULL")
        self.assertEqual(lookup.loc[("admission_type_id", "6"), "lookup_status"], "Unclear description")
        self.assertEqual(lookup.loc[("admission_type_id", "99"), "description"], "")
        self.assertEqual(lookup.loc[("admission_type_id", "99"), "lookup_status"], "Unmapped code")
        self.assertEqual(lookup.loc[("admission_source_id", "7"), "lookup_status"], "Mapped")


class CohortInputTests(TestCase):
    def test_rejects_wrong_patient_for_verified_encounter(self) -> None:
        key = COHORT_DATA_PATH.relative_to(PROJECT_ROOT).as_posix()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = pd.DataFrame(
                [{"encounter_id": "001", "patient_nbr": "101"}], columns=ENCOUNTER_COLUMNS, dtype="string",
            ).fillna("")
            path = write_csv(root / key, source)
            manifest = {"artifact_sha256": {key: sha256_file(path)}}
            prepared = pd.DataFrame({"encounter_id": ["001"], "patient_nbr": ["102"]})
            with self.assertRaisesRegex(ValueError, "Patient ID differs"):
                _load_source_cohort(root, manifest, prepared)


class EdaExportTests(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        missing = [str(path.relative_to(PROJECT_ROOT)) for path in INPUT_PATHS.values() if not path.is_file()]
        if missing:
            raise SkipTest("EDA export requires prepared inputs. Missing: " + ", ".join(missing))
        cls.output_dir = Path(cls.enterClassContext(TemporaryDirectory(prefix="eda-validation-")))
        export_eda(cls.output_dir)

    def test_manifest_matches_all_sources_and_outputs(self) -> None:
        manifest = read_json(self.output_dir / MANIFEST_PATH)
        self.assertEqual(manifest["scope"]["patients"], 69_990)
        self.assertEqual(manifest["runtime"]["matplotlib"], version("matplotlib"))
        self.assertEqual(manifest["inputs"].keys(), INPUT_PATHS.keys())
        for name, record in manifest["inputs"].items():
            self.assertEqual(record["path"], INPUT_PATHS[name].relative_to(PROJECT_ROOT).as_posix())
            content = (PROJECT_ROOT / record["path"]).read_bytes()
            if name in {"preparation_config", "target_config"}:
                self.assertEqual(record["checksum_mode"], "LF line endings")
                content = content.replace(b"\r\n", b"\n")
            else:
                self.assertEqual(record["checksum_mode"], "exact file bytes")
            self.assertEqual(record["sha256"], sha256(content).hexdigest())

        files = {
            path.relative_to(self.output_dir).as_posix(): sha256_file(path)
            for path in self.output_dir.rglob("*") if path.is_file()
        }
        files.pop(MANIFEST_PATH.as_posix())
        self.assertEqual(files, manifest["artifact_sha256"])
        self.assertEqual(Counter(Path(path).suffix for path in files), {".csv": 28, ".png": 10})
        for relative in files:
            if relative.endswith(".csv"):
                columns = read_csv(self.output_dir / relative).columns
                self.assertTrue({"encounter_id", "patient_nbr"}.isdisjoint(columns), relative)

    def test_exported_evidence_reconciles_across_tables(self) -> None:
        def table(name: str) -> list[dict[str, str]]:
            return read_csv(self.output_dir / TABLE_DIR / f"{name}.csv").to_dict(orient="records")

        dictionary = table("column_dictionary")
        self.assertEqual(Counter(row["role"] for row in dictionary), {
            "identifier": 2, "predictor": 32, "target": 1,
        })
        nominal_totals = Counter()
        for row in table("nominal_frequency"):
            nominal_totals[row["column"]] += int(row["patients"])
        self.assertEqual(len(nominal_totals), 24)
        self.assertEqual(set(nominal_totals.values()), {69_990})
        self.assertEqual(sum(int(row["empty_fields"]) for row in table("missing_summary")), 1_922)

        for row in table("laboratory_results"):
            if row["value"] == "None":
                self.assertEqual(row["percent_of_recorded"], "")
                self.assertEqual(row["result_recorded"], "False")
        outcome = {(row["representation"], row["label"]): int(row["patients"]) for row in table("outcome_distribution")}
        self.assertEqual(outcome["Any Readmission", "readmitted"], outcome["Original", "<30"] + outcome["Original", ">30"])
        findings = {row["finding"]: row for row in table("eda_findings")}
        self.assertIn("28,511/69,990", findings["Any Readmission"]["evidence"])
        self.assertIn("54,041/69,990", findings["Laboratory coverage"]["evidence"])

    def test_cohort_selection_counts_reconcile(self) -> None:
        flow = read_csv(self.output_dir / TABLE_DIR / "cohort_flow.csv")
        self.assertEqual(flow["encounters"].astype(int).tolist(), [101_766, 99_343, 69_990])

        comparisons = (
            ("discharge_eligibility_profile", {"eligible": 99_343, "excluded": 2_423}),
            ("representative_encounter_profile", {"representative": 16_341, "other_eligible": 29_353}),
        )
        for name, expected in comparisons:
            with self.subTest(table=name):
                frame = read_csv(self.output_dir / TABLE_DIR / f"{name}.csv")
                actual = frame.groupby(["group", "attribute"])
                self.assertEqual(set(frame["group"]), set(expected))
                self.assertEqual(set(frame["attribute"]), {"age", "time_in_hospital", "admission_type_id"})
                self.assertEqual(actual.ngroups, 6)
                for (group, _), rows in actual:
                    self.assertEqual(rows["encounters"].astype(int).sum(), expected[group])
                    self.assertEqual(set(rows["group_encounters"].astype(int)), {expected[group]})
                    self.assertAlmostEqual(rows["percent_of_group"].astype(float).sum(), 100)

        distribution = read_csv(self.output_dir / TABLE_DIR / "eligible_encounters_per_patient.csv")
        multiple = distribution.loc[distribution["eligible_encounters_per_patient"].astype(int).gt(1)]
        self.assertEqual(distribution["patients"].astype(int).sum(), 69_990)
        self.assertEqual(multiple["patients"].astype(int).sum(), 16_341)
        self.assertEqual(multiple["eligible_encounters"].astype(int).sum(), 45_694)

        overlap = read_csv(self.output_dir / TABLE_DIR / "discharge_patient_overlap.csv").set_index("group")
        self.assertEqual(overlap["patients"].astype(int).to_dict(), {
            "eligible_only": 69_119, "eligible_and_excluded": 871, "excluded_only": 1_528,
        })
        self.assertEqual(overlap["patients"].astype(int).sum(), 71_518)
