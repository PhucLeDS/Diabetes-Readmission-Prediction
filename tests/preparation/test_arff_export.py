"""Verify Weka schema fitting, ARFF conversion, and read-back validation."""

from unittest import TestCase

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
        self.train = [{
            "admission_source_id": "1",
            "number_inpatient": "2",
            "A1Cresult": "None",
            "diag_1_group": "Missing",
            CLASS_COLUMN: "not_readmitted",
        }]
        self.validation = [self.train[0] | {
            "admission_source_id": "14",
            "number_inpatient": "",
            CLASS_COLUMN: "readmitted",
        }]

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
        final = self.schema(self.train + self.validation)
        self.assertEqual(final[0].domain, ("1", "14"))
        with self.assertRaisesRegex(ValueError, "Unseen value"):
            make_arff("model", development, self.validation)
        source = [self.validation[0].copy()]
        self.validation[0]["number_inpatient"] = "?"
        source[0]["number_inpatient"] = "?"

        text, audit = make_arff("model", development, self.validation, allow_unseen=True)
        validate_arff(text, "model", development, self.validation, allow_unseen=True)
        self.assertTrue(text.endswith("?,?,'None','Missing','readmitted'\n"))
        self.assertEqual(audit["source_missing_by_column"], {"number_inpatient": 1})
        self.assertEqual(audit["unseen_nominal_values_by_column"], {"admission_source_id": {"14": 1}})
        text, audit = make_arff("model", final, self.validation, allow_unseen=True)
        validate_arff(text, "model", final, self.validation, allow_unseen=True)
        self.assertTrue(text.endswith("'14',?,'None','Missing','readmitted'\n"))
        self.assertEqual(audit["unseen_nominal_values_by_column"], {})
        self.assertEqual(self.validation, source)

    def test_quoted_literal_round_trip(self):
        value = "O'Reilly, 50% \"test\" \\line\n\r\t\b\f"
        rows = [{"arbitrary field": value, CLASS_COLUMN: "readmitted"}]
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
                make_arff("model", schema, [self.train[0] | {CLASS_COLUMN: value}], allow_unseen=True)
        for value in ("True", "NaN", "1e999"):
            with self.subTest(numeric=value), self.assertRaisesRegex(ValueError, "Invalid numeric"):
                make_arff("model", schema, [self.train[0] | {"number_inpatient": value}])

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
