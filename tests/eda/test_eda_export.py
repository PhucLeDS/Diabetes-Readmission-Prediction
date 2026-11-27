"""Check the aggregate EDA bundle."""

from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import SkipTest, TestCase
from unittest.mock import patch

import pandas as pd

from src.eda.admission_context import profile_admission_context
from src.eda.data import _admission_descriptions, _load_source_cohort
from src.eda.diagnosis_detail import profile_diagnosis_detail
from src.eda.export import INPUT_PATHS, MANIFEST_PATH, TABLE_DIR, export_eda
from src.eda.information_coverage import profile_information_coverage
from src.eda.nominal import _diagnosis_summaries
from src.io_utils import read_csv, read_json, sha256_file, write_csv
from src.paths import PROJECT_ROOT
from src.preparation.data_loading import COUNT_COLUMNS, ENCOUNTER_COLUMNS, MAPPING_COLUMNS, RawTable
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


class InformationCoverageTests(TestCase):
    def test_missing_counts_and_invalid_laboratory_fields_remain_unknown(self) -> None:
        cohort = pd.DataFrame({
            "number_outpatient": pd.array([0, 1, 1, 0, 0, 0], dtype="Int64"),
            "number_emergency": pd.array([0, 0, pd.NA, 0, 0, 0], dtype="Int64"),
            "number_inpatient": pd.array([0, 0, 1, 0, 0, 0], dtype="Int64"),
            "A1Cresult": ["None", "Norm", "None", "", "Unexpected", "None"],
            "max_glu_serum": ["None", "None", ">200", "None", "Norm", pd.NA],
            "age": ["[20-30)"] * 6,
            "admission_type_id": ["1"] * 6,
            "admission_source_id": ["7"] * 6,
            "time_in_hospital": pd.array([1, 1, 1, 1, 1, 1], dtype="Int64"),
        })
        coverage = profile_information_coverage(cohort)
        history = coverage.lab_recording_by_prior_history.set_index(["prior_history", "lab_recording_pattern"])
        self.assertEqual(history.loc[history["patients"].gt(0), "patients"].to_dict(), {
            ("no_recorded_prior_visits", "neither"): 1,
            ("no_recorded_prior_visits", "unknown"): 3,
            ("any_recorded_prior_visit", "a1c_only"): 1,
            ("unknown", "glucose_only"): 1,
        })
        rows = coverage.lab_recording_by_characteristic
        by_stay = rows.loc[rows["characteristic"].eq("time_in_hospital")].set_index("lab_recording_pattern")
        self.assertEqual(by_stay.at["unknown", "patients"], 3)
        self.assertEqual(by_stay.at["unknown", "percent_of_group"], 50)


class AdmissionContextTests(TestCase):
    def test_blank_codes_and_missing_counts_keep_their_denominators(self) -> None:
        cohort = pd.DataFrame({
            "admission_type_id": ["1", "1", ""],
            "admission_source_id": ["7", "99", "7"],
            "age": ["[20-30)", "[30-40)", ""],
            **{column: pd.array([0, 2, pd.NA], dtype="Int64") for column in COUNT_COLUMNS},
        })
        descriptions = pd.DataFrame([
            ("admission_type_id", "1", "Emergency", "Mapped"),
            ("admission_type_id", "", "", "Empty CSV field"),
            ("admission_source_id", "7", "Emergency Room", "Mapped"),
            ("admission_source_id", "99", "", "Unmapped code"),
        ], columns=["column", "code", "description", "lookup_status"])
        admission = profile_admission_context(cohort, descriptions)
        pairs = admission.admission_type_source.set_index(["admission_type_code", "admission_source_code"])
        self.assertEqual(pairs.at[("1", "99"), "admission_source_lookup_status"], "Unmapped code")
        self.assertEqual(pairs.at[("", "7"), "admission_type_lookup_status"], "Empty CSV field")
        self.assertEqual(pairs.at[("", "99"), "patients"], 0)
        self.assertEqual(pairs.at[("1", "99"), "percent_of_admission_type"], 50)

        numeric = admission.numeric_by_admission.set_index(["column", "code", "measure"])
        source = numeric.loc[("admission_source_id", "7", "number_inpatient")]
        self.assertEqual((source["group_patients"], source["valid"], source["missing"]), (2, 1, 1))
        self.assertEqual(source["zero_percent_of_valid"], 100)
        no_valid = numeric.loc[("admission_type_id", "", "number_inpatient")]
        self.assertTrue(pd.isna(no_valid["zero_percent_of_valid"]))
        self.assertTrue(pd.isna(no_valid["median"]))


class DiagnosisDetailTests(TestCase):
    def setUp(self) -> None:
        self.cohort = pd.DataFrame({
            "encounter_id": ["001", "002", "003", "004"],
            "patient_nbr": ["101", "102", "103", "104"],
            "diag_1_group": ["Circulatory", "Other_V", "Missing", "Other_E"],
            "diag_2_group": ["Circulatory", "Circulatory", "Unmapped", "Missing"],
            "diag_3_group": ["Other_V", "Circulatory", "Missing", "Missing"],
        })
        self.source = pd.DataFrame({
            "encounter_id": ["002", "001", "003", "004"],
            "patient_nbr": ["102", "101", "103", "104"],
            "diag_1": ["V45.81", "401.9", "", "E812.0"],
            "diag_2": ["401", "401.9", "bad", ""],
            "diag_3": ["401.9", "V45.81", "", ""],
        })
        self.summary, _ = _diagnosis_summaries(self.cohort)

    def test_repeated_groups_pairs_and_source_codes_count_each_patient_once(self) -> None:
        diagnoses = profile_diagnosis_detail(self.cohort, self.source, self.summary)
        depth = diagnoses.diagnosis_position_depth.set_index([
            "mapped_positions", "distinct_mapped_groups", "missing_positions", "unmapped_positions",
        ])
        self.assertEqual(depth.at[(3, 2, 0, 0), "patients"], 2)
        self.assertEqual(depth.at[(0, 0, 2, 1), "patients"], 1)
        self.assertEqual(diagnoses.diagnosis_group_pairs.set_index(["group_1", "group_2"])["patients"].to_dict(), {
            ("Circulatory", "Other_V"): 2,
        })
        counts = diagnoses.diagnosis_source_code_detail.set_index(["group", "source_code"])
        self.assertEqual(counts.at[("Circulatory", "401.9"), "patients"], 2)
        self.assertEqual(counts.at[("Circulatory", "401"), "percent_of_group"], 50)
        self.assertEqual(counts.at[("Other_E", "E812.0"), "patients"], 1)
        self.assertEqual(counts.at[("Unmapped", "bad"), "patients"], 1)
        self.assertNotIn("", counts.index.get_level_values("source_code"))

    def test_rejects_mismatched_keys_or_group_assignments(self) -> None:
        wrong_patient = self.source.copy()
        wrong_patient.loc[0, "patient_nbr"] = "999"
        with self.assertRaisesRegex(ValueError, "Patient IDs differ"):
            profile_diagnosis_detail(self.cohort, wrong_patient, self.summary)
        with self.assertRaisesRegex(ValueError, "encounter IDs differ"):
            profile_diagnosis_detail(self.cohort, self.source.iloc[1:], self.summary)
        wrong_code = self.source.copy()
        wrong_code.loc[0, "diag_1"] = "401.9"
        with self.assertRaisesRegex(ValueError, "disagrees with its prepared diagnosis group"):
            profile_diagnosis_detail(self.cohort, wrong_code, self.summary)


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
        self.assertEqual(Counter(Path(path).suffix for path in files), {".csv": 38, ".png": 14})
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

    def test_information_coverage_reconciles_with_overall_recording_patterns(self) -> None:
        history = read_csv(self.output_dir / TABLE_DIR / "lab_recording_by_prior_history.csv")
        characteristics = read_csv(self.output_dir / TABLE_DIR / "lab_recording_by_characteristic.csv")
        patterns = read_csv(self.output_dir / TABLE_DIR / "lab_recording_patterns.csv")
        status = {
            ("False", "False"): "neither", ("True", "False"): "a1c_only",
            ("False", "True"): "glucose_only", ("True", "True"): "both",
        }
        expected = {
            status[(row.a1c_result_recorded, row.glucose_result_recorded)]: int(row.patients)
            for row in patterns.itertuples(index=False)
        }
        expected["unknown"] = 0
        self.assertEqual(sum(expected.values()), 69_990)
        self.assertEqual(history.groupby("lab_recording_pattern")["patients"].agg(
            lambda counts: counts.astype(int).sum(),
        ).to_dict(), expected)
        self.assertEqual(set(characteristics["characteristic"]), {
            "age", "admission_type_id", "admission_source_id", "time_in_hospital",
        })
        for _, rows in characteristics.groupby("characteristic"):
            totals = rows.assign(patients=rows["patients"].astype(int)).groupby("lab_recording_pattern")["patients"]
            self.assertEqual(totals.sum().to_dict(), expected)

        combined = history.set_index(["prior_history", "lab_recording_pattern"])
        self.assertEqual(int(combined.at[("no_recorded_prior_visits", "neither"), "patients"]), 39_726)
        for frame, keys in ((history, "prior_history"), (characteristics, ["characteristic", "value"])):
            for _, rows in frame.groupby(keys):
                counts = rows["patients"].astype(int)
                denominator = counts.sum()
                self.assertEqual(set(rows["group_patients"].astype(int)), {denominator})
                self.assertEqual(rows["small_cell"].eq("True").tolist(), counts.lt(30).tolist())
                if denominator:
                    for count, percent in zip(counts, rows["percent_of_group"].astype(float), strict=True):
                        self.assertAlmostEqual(percent, 100 * count / denominator)
                else:
                    self.assertTrue(rows["percent_of_group"].eq("").all())

    def test_admission_tables_reconcile_with_nominal_profiles(self) -> None:
        def read_table(name: str) -> pd.DataFrame:
            return read_csv(self.output_dir / TABLE_DIR / f"{name}.csv")

        nominal = read_table("admission_descriptions")
        pairs = read_table("admission_type_source")
        pairs["patients"] = pairs["patients"].astype(int)
        ages = read_table("age_by_admission")
        numeric = read_table("numeric_by_admission")

        for column, key in (("admission_type_id", "admission_type_code"), ("admission_source_id", "admission_source_code")):
            labels = nominal.loc[nominal["column"].eq(column)].set_index("code")
            expected = labels["patients"].astype(int).to_dict()
            self.assertEqual(pairs.groupby(key)["patients"].sum().to_dict(), expected)
            age_rows = ages.loc[ages["column"].eq(column)].assign(patients=lambda rows: rows["patients"].astype(int))
            self.assertEqual(age_rows.groupby("code")["patients"].sum().to_dict(), expected)
            summaries = numeric.loc[numeric["column"].eq(column)]
            self.assertEqual(set(summaries["measure"]), set(COUNT_COLUMNS))
            for _, rows in summaries.groupby("measure"):
                self.assertEqual(rows.set_index("code")["group_patients"].astype(int).to_dict(), expected)
                self.assertEqual(
                    (rows["valid"].astype(int) + rows["missing"].astype(int)).tolist(),
                    rows["group_patients"].astype(int).tolist(),
                )
            status = pairs[[key, key.replace("_code", "_lookup_status")]].drop_duplicates().set_index(key)
            self.assertEqual(status.iloc[:, 0].to_dict(), labels["lookup_status"].to_dict())

        for _, rows in pairs.groupby("admission_type_code"):
            self.assertEqual(set(rows["type_patients"].astype(int)), {rows["patients"].sum()})
            for count, share in zip(rows["patients"], rows["percent_of_admission_type"].astype(float), strict=True):
                self.assertAlmostEqual(share, 100 * count / rows["patients"].sum())

    def test_diagnosis_depth_reconciles_with_nominal_coverage(self) -> None:
        depth = read_csv(self.output_dir / TABLE_DIR / "diagnosis_position_depth.csv").astype({
            "mapped_positions": int, "distinct_mapped_groups": int, "patients": int,
            "missing_positions": int, "unmapped_positions": int,
        })
        complete = depth.loc[depth["mapped_positions"].eq(3)].set_index("distinct_mapped_groups")
        self.assertEqual(complete["patients"].to_dict(), {1: 7_360, 2: 31_070, 3: 30_276})
        coverage = read_csv(self.output_dir / TABLE_DIR / "diagnosis_coverage.csv")
        nominal_coverage = coverage.set_index(coverage["positions_with_mapped_codes"].astype(int))
        totals = depth.groupby("mapped_positions")["patients"].sum()
        self.assertEqual(totals.reindex(nominal_coverage.index, fill_value=0).to_dict(),
                         nominal_coverage["patients"].astype(int).to_dict())
        nominal = read_csv(self.output_dir / TABLE_DIR / "diagnosis_summary.csv").set_index("group")
        for status in ("Missing", "Unmapped"):
            positions = [f"diag_{n}_group_patients" for n in (1, 2, 3)]
            expected = nominal.loc[status, positions].astype(int).sum()
            actual = (depth[f"{status.lower()}_positions"] * depth["patients"]).sum()
            self.assertEqual(actual, expected)

        gain = read_csv(self.output_dir / TABLE_DIR / "diagnosis_group_gain.csv").set_index("group")
        for row in gain.itertuples():
            expected = nominal.loc[row.Index]
            self.assertEqual(row.patients_at_primary_position, expected["diag_1_group_patients"])
            self.assertEqual(row.patients_at_any_position, expected["patients_at_any_position"])
            self.assertEqual(int(row.additional_patients),
                             int(row.patients_at_any_position) - int(row.patients_at_primary_position))
