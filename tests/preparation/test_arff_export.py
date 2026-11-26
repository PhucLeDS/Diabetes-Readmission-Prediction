"""Verify Weka schema fitting, ARFF conversion, and read-back validation."""

from unittest import TestCase

import pandas as pd

from src.preparation.dataset_export import (
    CLASS_COLUMN,
    CLASS_LABELS,
    ArffAttribute,
    arff_header,
    create_arff_schema,
    make_arff,
    quote_arff,
    validate_arff,
)


class ArffExportTests(TestCase):
    def setUp(self):
        self.columns = ("admission_source_id", "number_inpatient", "A1Cresult", "diag_1_group")
        train = {
            "admission_source_id": "1",
            "number_inpatient": "2",
            "A1Cresult": "None",
            "diag_1_group": "Missing",
            CLASS_COLUMN: "not_readmitted",
        }
        self.train = pd.DataFrame([train], dtype="string")
        self.validation = pd.DataFrame([train | {
            "admission_source_id": "14",
            "number_inpatient": "",
            CLASS_COLUMN: "readmitted",
        }], dtype="string")

    def schema(self, training_rows):
        return create_arff_schema(
            self.columns, training_rows, target=CLASS_COLUMN, target_values=CLASS_LABELS,
        )

    def test_fixed_types_and_literal_categories(self):
        schema = self.schema(self.train)
        self.assertEqual(schema, (
            ArffAttribute("admission_source_id", ("1",)),
            ArffAttribute("number_inpatient"),
            ArffAttribute("A1Cresult", ("None",)),
            ArffAttribute("diag_1_group", ("Missing",)),
            ArffAttribute(CLASS_COLUMN, CLASS_LABELS),
        ))
        content, audit = make_arff("development", schema, self.train)
        self.assertTrue(content.endswith("'1',2,'None','Missing','not_readmitted'\n"))
        self.assertEqual(audit["arff_missing_by_column"], {})
        validate_arff(content, "development", schema, self.train)

    def test_unseen_category_depends_on_fitting_scope(self):
        development = self.schema(self.train)
        final = self.schema(pd.concat([self.train, self.validation], ignore_index=True))
        self.assertEqual(final[0].domain, ("1", "14"))
        with self.assertRaisesRegex(ValueError, "Unseen value"):
            make_arff("model", development, self.validation)
        self.validation.at[0, "number_inpatient"] = "?"
        source = self.validation.copy()

        text, audit = make_arff("model", development, self.validation, allow_unseen=True)
        validate_arff(text, "model", development, self.validation, allow_unseen=True)
        self.assertTrue(text.endswith("?,?,'None','Missing','readmitted'\n"))
        self.assertEqual(audit["source_missing_by_column"], {"number_inpatient": 1})
        self.assertEqual(audit["unseen_nominal_values_by_column"], {"admission_source_id": {"14": 1}})
        text, audit = make_arff("model", final, self.validation, allow_unseen=True)
        validate_arff(text, "model", final, self.validation, allow_unseen=True)
        self.assertTrue(text.endswith("'14',?,'None','Missing','readmitted'\n"))
        self.assertEqual(audit["unseen_nominal_values_by_column"], {})
        pd.testing.assert_frame_equal(self.validation, source)

    def test_quoted_literal_round_trip(self):
        value = "O'Reilly, 50% \"test\" \\line\n\r\t\b\f"
        rows = pd.DataFrame([{"arbitrary field": value, CLASS_COLUMN: "readmitted"}], dtype="string")
        schema = create_arff_schema(("arbitrary field",), rows, target=CLASS_COLUMN, target_values=CLASS_LABELS)
        content, _ = make_arff("rel,ation", schema, rows)
        self.assertIn(r"'O\'Reilly, 50\% \"test\" \\line\n\r\t\b\f'", content)
        validate_arff(content, "rel,ation", schema, rows)
        with self.assertRaisesRegex(ValueError, "contain NUL"):
            quote_arff("\x00")

    def test_rejects_missing_class_and_invalid_numeric_values(self):
        schema = self.schema(self.train)
        for value in ("", "?", "NO"):
            with self.subTest(target=value), self.assertRaisesRegex(ValueError, "Invalid ARFF target"):
                make_arff("model", schema, self.train.assign(**{CLASS_COLUMN: value}), allow_unseen=True)
        for value in ("True", "NaN", "1e999"):
            with self.subTest(numeric=value), self.assertRaisesRegex(ValueError, "Invalid numeric"):
                make_arff("model", schema, self.train.assign(number_inpatient=value))

    def test_read_back_rejects_mismatched_relation_row_and_type(self):
        schema = self.schema(self.train)
        content, _ = make_arff("model", schema, self.train)
        with self.assertRaisesRegex(ValueError, "ordered attribute schema"):
            validate_arff(content, "wrong", schema, self.train)
        for row in ("'1',3,'None','Missing','not_readmitted'", "1,2,'None','Missing','not_readmitted'"):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "values differ"):
                validate_arff(arff_header("model", schema) + row + "\n", "model", schema, self.train)
        with self.assertRaises(ValueError):
            validate_arff(content + content.split("@DATA\n")[1], "model", schema, self.train)
