"""Check the aggregate EDA bundle."""

from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import SkipTest, TestCase

from src.eda.export import INPUT_PATHS, MANIFEST_PATH, TABLE_DIR, export_eda
from src.io_utils import read_csv, read_json, sha256_file
from src.paths import PROJECT_ROOT


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
        self.assertEqual(Counter(Path(path).suffix for path in files), {".csv": 24, ".png": 10})
        for relative in files:
            if relative.endswith(".csv"):
                columns, _ = read_csv(self.output_dir / relative)
                self.assertTrue({"encounter_id", "patient_nbr"}.isdisjoint(columns), relative)

    def test_exported_evidence_reconciles_across_tables(self) -> None:
        def table(name: str) -> list[dict[str, str]]:
            return read_csv(self.output_dir / TABLE_DIR / f"{name}.csv")[1]

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
