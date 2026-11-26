"""Tests for the shared project utilities."""

import codecs
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import io_utils, paths, validation


class PathTests(unittest.TestCase):
    def test_resolves_project_paths_from_notebooks_directory(self):
        root = next(
            parent for parent in Path(__file__).resolve().parents
            if (parent / "src" / "paths.py").is_file() and (parent / "requirements.txt").is_file()
        )
        environment = dict(os.environ, PYTHONPATH=str(root))
        result = subprocess.run(
            [
                sys.executable, "-B", "-c",
                "from src.paths import PROJECT_ROOT, RAW_DATA_PATH;"
                "print(PROJECT_ROOT); print(RAW_DATA_PATH)",
            ],
            cwd=root / "notebooks", env=environment, check=True, capture_output=True,
            encoding="utf-8",
        )
        self.assertEqual(result.stdout.splitlines(), [
            str(root), str(root / "data" / "raw" / "diabetic_data.csv"),
        ])
        self.assertEqual(paths.PREPARATION_OUTPUT_DIR, root / "outputs" / "preparation")


class FileTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_csv_preserves_literals_identifiers_and_embedded_newlines(self):
        target = self.directory / "nested" / "table.csv"
        columns = ["encounter", "result", "comment", "count"]
        rows = [
            {"encounter": "0001", "result": "None", "comment": "a\rb", "count": 2},
            {"encounter": "0002", "result": "?", "comment": "a,b\nc", "count": None},
        ]
        self.assertEqual(io_utils.write_csv(target, columns, iter(rows)), target)
        self.assertEqual(
            target.read_bytes(),
            b'encounter,result,comment,count\n0001,None,"a\rb",2\n0002,?,"a,b\nc",\n',
        )
        header, values = io_utils.read_csv(target, expected_columns=columns)
        self.assertEqual(header, columns)
        self.assertEqual(values, [
            {"encounter": "0001", "result": "None", "comment": "a\rb", "count": "2"},
            {"encounter": "0002", "result": "?", "comment": "a,b\nc", "count": ""},
        ])
        frame = io_utils.read_csv_frame(target, expected_columns=columns)
        pd.testing.assert_frame_equal(frame, pd.DataFrame(values, columns=columns, dtype="string"))
        self.assertEqual(io_utils.write_csv_frame(target, frame.set_axis([7, 8])), target)

    def test_csv_reads_byte_order_mark_and_windows_newlines(self):
        target = self.directory / "input.csv"
        target.write_bytes(codecs.BOM_UTF8 + b"id,value\r\n001,None\r\n")
        self.assertEqual(io_utils.read_csv(target), (["id", "value"], [{"id": "001", "value": "None"}]))
        frame = io_utils.read_csv_frame(target)
        self.assertEqual(frame.iloc[0].to_dict(), {"id": "001", "value": "None"})

    def test_csv_rejects_invalid_headers_and_row_widths(self):
        target = self.directory / "invalid.csv"
        for content, error in (
            (b"", "nonempty column names"),
            (b"id,id\n1,2\n", "unique column names"),
            (b"id,\n1,2\n", "nonempty column names"),
            (b"id,value\n1\n", "field count"),
            (b"id,value\n1,2,3\n", "field count"),
            (b"value,id\n2,1\n", "column order"),
        ):
            with self.subTest(content=content):
                target.write_bytes(content)
                for reader in (io_utils.read_csv_frame, io_utils.read_csv):
                    with self.assertRaisesRegex(ValueError, error):
                        reader(target, expected_columns=["id", "value"])

    def test_failed_csv_write_keeps_previous_output(self):
        target = self.directory / "output.csv"
        io_utils.write_csv(target, ["id"], [{"id": "001"}])
        with self.assertRaisesRegex(ValueError, "data row 2"):
            io_utils.write_csv(target, ["id"], [{"id": "002"}, {"wrong": "003"}], overwrite=True)
        self.assertEqual(target.read_bytes(), b"id\n001\n")
        self.assertEqual(list(self.directory.iterdir()), [target])

    def test_json_round_trip_and_checksum(self):
        target = self.directory / "manifest.json"
        expected = {"z": "\u03b1", "metadata": {"rows": 2, "complete": True}}
        io_utils.write_json(target, expected)
        payload = target.read_bytes()
        self.assertTrue(payload.startswith(b'{\n  "metadata"'))
        self.assertIn("\u03b1".encode("utf-8"), payload)
        self.assertEqual(io_utils.read_json(target), expected)
        self.assertEqual(io_utils.sha256_file(target), hashlib.sha256(payload).hexdigest())

    def test_retries_are_idempotent_and_conflicting_writes_require_overwrite(self):
        target = self.directory / "manifest.json"
        io_utils.write_json(target, {"a": 1, "b": 2})
        original = target.read_bytes()
        initial_mtime = target.stat().st_mtime_ns

        io_utils.write_json(target, {"b": 2, "a": 1})
        self.assertEqual(target.stat().st_mtime_ns, initial_mtime)
        with self.assertRaisesRegex(FileExistsError, "Existing output differs"):
            io_utils.write_json(target, {"a": 3})
        self.assertEqual(target.read_bytes(), original)
        io_utils.write_json(target, {"a": 3}, overwrite=True)
        self.assertEqual(io_utils.read_json(target), {"a": 3})
        self.assertEqual(list(self.directory.iterdir()), [target])

    def test_invalid_json_write_does_not_publish_an_artifact(self):
        target = self.directory / "manifest.json"
        with self.assertRaises(ValueError):
            io_utils.write_json(target, {"missing": float("nan")})
        self.assertFalse(target.exists())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_artifact_conflict_is_checked_before_any_write(self):
        existing = self.directory / "existing.png"
        new = self.directory / "tables" / "new.csv"
        existing.write_bytes(b"old image")
        artifacts = {
            new.relative_to(self.directory): b"value\n1\n",
            existing.relative_to(self.directory): b"new image",
        }
        with self.assertRaisesRegex(FileExistsError, "existing.png"):
            io_utils.write_artifacts(self.directory, artifacts)
        self.assertEqual(existing.read_bytes(), b"old image")
        self.assertFalse(new.parent.exists())

        io_utils.write_artifacts(self.directory, artifacts, overwrite=True)
        self.assertEqual(existing.read_bytes(), b"new image")
        self.assertEqual(new.read_bytes(), b"value\n1\n")

        with patch.object(io_utils, "write_bytes") as writer:
            io_utils.write_artifacts(self.directory, artifacts)
            io_utils.write_artifacts(self.directory, artifacts, overwrite=True)
            writer.assert_not_called()

    def test_artifact_retry_completes_partial_export(self):
        data = self.directory / "table.csv"
        manifest = self.directory / "manifest.json"
        artifacts = {Path("table.csv"): b"value\n1\n", Path("manifest.json"): b"{}\n"}
        write_bytes = io_utils.write_bytes

        def fail_on_manifest(path, content, *, overwrite):
            if path == manifest:
                raise OSError("Simulated interruption")
            return write_bytes(path, content, overwrite=overwrite)

        with patch.object(io_utils, "write_bytes", side_effect=fail_on_manifest):
            with self.assertRaisesRegex(OSError, "Simulated interruption"):
                io_utils.write_artifacts(self.directory, artifacts)
        self.assertEqual(data.read_bytes(), b"value\n1\n")
        self.assertFalse(manifest.exists())

        with patch.object(io_utils, "write_bytes", wraps=write_bytes) as writer:
            io_utils.write_artifacts(self.directory, artifacts)
            writer.assert_called_once_with(manifest, b"{}\n", overwrite=False)
        self.assertEqual(manifest.read_bytes(), b"{}\n")


class ValidationTests(unittest.TestCase):
    def test_required_columns_accept_extra_columns(self):
        validation.validate_required_columns(["id", "label", "feature"], ["id", "label"])
        with self.assertRaisesRegex(ValueError, "Missing required columns: label"):
            validation.validate_required_columns(["id"], ["id", "label"])

    def test_parsed_types_respect_nullability_and_reject_boolean_counts(self):
        schema = {"id": str, "count": int}
        validation.validate_types([{"id": "001", "count": None}], schema, nullable={"count"})
        with self.assertRaisesRegex(ValueError, "expected int, got bool"):
            validation.validate_types([{"id": "001", "count": True}], schema)
        with self.assertRaisesRegex(ValueError, "expected int, got NoneType"):
            validation.validate_types([{"id": "001", "count": None}], schema)

    def test_missing_values_are_column_scoped_and_configurable(self):
        rows = [{"id": "001", "result": "None", "unused": ""}]
        validation.validate_no_missing(rows, ["id", "result"])
        with self.assertRaisesRegex(ValueError, "Missing value.*'unused'"):
            validation.validate_no_missing(rows, ["unused"])
        with self.assertRaisesRegex(ValueError, "Missing value.*'result'"):
            validation.validate_no_missing([{"result": "?"}], ["result"], missing_values={"", "?"})
        with self.assertRaisesRegex(ValueError, "has no column 'other'"):
            validation.validate_no_missing(rows, ["other"])

    def test_ids_and_target_domains_are_checked_without_coercion(self):
        validation.validate_unique_ids(["001", "1"], name="encounter ID")
        with self.assertRaisesRegex(ValueError, "Duplicate encounter ID"):
            validation.validate_unique_ids(["001", "001"], name="encounter ID")
        with self.assertRaisesRegex(ValueError, "Invalid ID"):
            validation.validate_unique_ids([""])
        validation.validate_target(["NO", "<30", ">30"], allowed={"NO", "<30", ">30"})
        with self.assertRaisesRegex(ValueError, "data row 2"):
            validation.validate_target(["NO", ""], allowed={"NO", "<30", ">30"})

    def test_compatible_schemas_require_attribute_order_and_nominal_domains(self):
        expected = {"age": "numeric", "class": ("not_readmitted", "readmitted")}
        validation.validate_compatible_schemas(expected.copy(), expected)
        with self.assertRaisesRegex(ValueError, "column order"):
            validation.validate_compatible_schemas(
                {"class": expected["class"], "age": "numeric"}, expected,
            )
        with self.assertRaisesRegex(ValueError, "Incompatible attribute definition for 'class'"):
            validation.validate_compatible_schemas(
                {"age": "numeric", "class": ("readmitted", "not_readmitted")}, expected,
            )

    def test_partitions_check_coverage_pairing_and_patient_isolation(self):
        source = [("01", "A"), ("02", "A"), ("03", "B")]
        validation.validate_partitions(source, {"train": source[:2], "test": source[2:]})
        cases = [
            ({"train": source, "test": [("01", "A")]}, "more than once"),
            ({"train": source[:1], "test": source[1:]}, "appears in both"),
            ({"train": source[:2]}, "1 source encounter"),
            ({"train": [("04", "A")]}, "Unexpected encounter"),
            ({"train": [("01", "B")]}, "Patient ID differs"),
        ]
        for partitions, error in cases:
            with self.subTest(partitions=partitions):
                with self.assertRaisesRegex(ValueError, error):
                    validation.validate_partitions(source, partitions)
        with self.assertRaisesRegex(ValueError, "Duplicate source encounter"):
            validation.validate_partitions([*source, source[0]], {"train": source})


if __name__ == "__main__":
    unittest.main()
