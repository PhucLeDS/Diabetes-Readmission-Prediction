"""Check artifact reconciliation and read-back for Analysis."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import SkipTest, TestCase

from src.analysis.export import INPUT_PATHS, MANIFEST_PATH, TABLE_DIR, export_analysis
from src.io_utils import read_csv, read_json, sha256_file
from src.paths import PROJECT_ROOT


class AnalysisExportTests(TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [str(path.relative_to(PROJECT_ROOT)) for path in INPUT_PATHS.values() if not path.is_file()]
        if missing:
            raise SkipTest("Analysis export requires prepared inputs. Missing: " + ", ".join(missing))
        cls.output_dir = Path(cls.enterClassContext(TemporaryDirectory(prefix="analysis-export-test-")))
        cls.manifest = read_json(export_analysis(cls.output_dir))

    def test_manifest_matches_the_inputs_and_all_exported_artifacts(self):
        self.assertTrue(self.manifest["scope"]["includes_test_patients"])
        self.assertEqual(self.manifest["scope"]["outcome_counts"], {"not_readmitted": 41479, "readmitted": 28511})
        self.assertEqual(self.manifest["inputs"].keys(), INPUT_PATHS.keys())
        for name, entry in self.manifest["inputs"].items():
            self.assertEqual(entry["path"], INPUT_PATHS[name].relative_to(PROJECT_ROOT).as_posix())
            content = (PROJECT_ROOT / entry["path"]).read_bytes()
            if name in {"analysis_config", "preparation_config", "target_config"}:
                self.assertEqual(entry["checksum_mode"], "LF line endings")
                content = content.replace(b"\r\n", b"\n")
            else:
                self.assertEqual(entry["checksum_mode"], "exact file bytes")
            self.assertEqual(entry["sha256"], sha256(content).hexdigest())

        files = {
            path.relative_to(self.output_dir): sha256_file(path)
            for path in self.output_dir.rglob("*") if path.is_file()
        }
        files.pop(MANIFEST_PATH)
        self.assertEqual(Counter(path.suffix for path in files), {".csv": 12, ".png": 3})
        self.assertEqual({path.as_posix(): digest for path, digest in files.items()}, self.manifest["artifact_sha256"])
        for path in (p for p in files if p.suffix == ".csv"):
            content = (self.output_dir / path).read_bytes()
            self.assertNotIn(b"\r", content)
            columns = read_csv(self.output_dir / path).columns
            self.assertTrue({"encounter_id", "patient_nbr"}.isdisjoint(columns))

    def test_csv_read_back_preserves_categories_and_evaluation_reference(self):
        def read_table(name):
            return read_csv(self.output_dir / TABLE_DIR / f"{name}.csv")

        labs = read_table("lab_results_by_outcome").to_dict(orient="records")
        none_counts = {row["column"]: int(row["patients"]) for row in labs if row["value"] == "None"}
        self.assertEqual(none_counts, {"A1Cresult": 57144, "max_glu_serum": 66641})
        self.assertTrue(all(not row["percent_of_recorded_results"] for row in labs if row["value"] == "None"))
        evaluation_table = read_table("analysis_evaluation_map")
        self.assertNotIn("status", evaluation_table.columns)
        evaluation = evaluation_table.to_dict(orient="records")
        self.assertEqual([row["question_id"] for row in evaluation], ["Q1", "Q2", "Q3", "Q4"])
        for row in evaluation:
            self.assertTrue(all((self.output_dir / TABLE_DIR / name).is_file()
                                for name in row["source_tables"].split(", ")))
