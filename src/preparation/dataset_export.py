"""Construct the binary modeling view and export reproducible cohort partitions."""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from io import StringIO
from math import floor, isfinite
from pathlib import Path

from src.io_utils import read_csv, read_json, sha256_file, write_csv, write_json, write_text
from src.paths import PREPARATION_OUTPUT_DIR, PROCESSED_DIR, PROJECT_ROOT, WEKA_DIR
from src.preparation.data_loading import COUNT_COLUMNS, ID_COLUMNS, TARGET_COLUMN, TARGET_LABELS
from src.preparation.dataset_preparation import (
    FULL_PREPARED_PATH,
    PREPARATION_CONFIG_PATH,
    PREPARATION_MANIFEST_PATH,
)
from src.validation import (
    validate_partitions,
    validate_required_columns,
    validate_schema,
    validate_target,
    validate_unique_ids,
)


TARGET_MAPPING_CONFIG_PATH = PROJECT_ROOT / "config" / "target_mapping.json"
CLASS_COLUMN = "class"
CLASS_LABELS = ("not_readmitted", "readmitted")


def load_full_prepared(*, project_root: Path = PROJECT_ROOT) -> tuple[list[str], list[dict[str, str]]]:
    """Read the verified Phase II CSV and retain its exact field values."""
    data_path = FULL_PREPARED_PATH.relative_to(PROJECT_ROOT)
    config_path = PREPARATION_CONFIG_PATH.relative_to(PROJECT_ROOT)
    manifest = read_json(project_root / PREPARATION_MANIFEST_PATH.relative_to(PROJECT_ROOT))
    csv_path = project_root / data_path
    configuration = manifest["configuration"]
    config_bytes = (project_root / config_path).read_bytes()
    config_lf = config_bytes.replace(b"\r\n", b"\n")
    # Git can convert the tracked configuration between LF and CRLF on checkout.
    config_checksums = {
        sha256(content).hexdigest()
        for content in (config_bytes, config_lf, config_lf.replace(b"\n", b"\r\n"))
    }
    if (
        configuration["path"] != config_path.as_posix()
        or configuration["sha256"] not in config_checksums
    ):
        raise ValueError("Preparation configuration differs from the manifest.")
    if (
        manifest["dataset_path"] != data_path.as_posix()
        or manifest["artifact_sha256"][data_path.as_posix()] != sha256_file(csv_path)
    ):
        raise ValueError("Full prepared CSV differs from the preparation manifest.")

    predictors = read_json(project_root / config_path)["feature_selection"]["predictors"]
    expected_columns = (*ID_COLUMNS, *predictors, TARGET_COLUMN)
    if len(predictors) != 32 or CLASS_COLUMN in predictors:
        raise ValueError("The preparation configuration must declare 32 predictors without the model class.")
    validate_schema([entry["name"] for entry in manifest["schema"]], expected=expected_columns)
    columns, rows = read_csv(csv_path, expected_columns=expected_columns)

    statistics = manifest["statistics"]
    if not rows or (len(rows), len(columns)) != (statistics["rows"], statistics["columns"]):
        raise ValueError("Full prepared CSV dimensions differ from the preparation manifest.")
    for column in ID_COLUMNS:
        validate_unique_ids((row[column] for row in rows), name=f"prepared {column}")
    validate_target((row[TARGET_COLUMN] for row in rows), allowed=TARGET_LABELS)
    counts = Counter(row[TARGET_COLUMN] for row in rows)
    if {label: counts[label] for label in TARGET_LABELS} != statistics["target_counts"]:
        raise ValueError("Original label counts differ from the preparation manifest.")
    return columns, rows


def load_target_mapping(config_path: Path = TARGET_MAPPING_CONFIG_PATH) -> dict[str, str]:
    """Require the configured source labels and their Any Readmission classes."""
    config = read_json(config_path)
    negative_class, positive_class = CLASS_LABELS
    mapping = config["class_mapping"]
    if (
        config["task_definition"] != "any_readmission"
        or config["target_source"] != TARGET_COLUMN
        or config["class_column"] != CLASS_COLUMN
        or config["class_values"] != list(CLASS_LABELS)
        or config["positive_class"] != positive_class
        or set(mapping) != set(TARGET_LABELS)
        or mapping["NO"] != negative_class
        or mapping["<30"] != positive_class
        or mapping[">30"] != positive_class
    ):
        raise ValueError("Target configuration must implement Any Readmission.")
    return mapping


def map_binary_targets(
    rows: Sequence[Mapping[str, str]], *, config_path: Path = TARGET_MAPPING_CONFIG_PATH,
) -> list[dict[str, str]]:
    """Append class to copies of the records while retaining identifiers and source labels."""
    mapping = load_target_mapping(config_path)
    validate_target((row.get(TARGET_COLUMN, "") for row in rows), allowed=mapping)
    if any(CLASS_COLUMN in row for row in rows):
        raise ValueError(f"Input records already contain the {CLASS_COLUMN!r} column.")
    return [{**row, CLASS_COLUMN: mapping[row[TARGET_COLUMN]]} for row in rows]


def build_model_dataset(
    prepared_columns: Sequence[str], mapped_rows: Sequence[Mapping[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """Project onto the ordered predictors followed by the binary class."""
    validate_schema(prepared_columns)
    if (
        len(prepared_columns) != 35
        or tuple(prepared_columns[:2]) != ID_COLUMNS
        or prepared_columns[-1] != TARGET_COLUMN
        or CLASS_COLUMN in prepared_columns
    ):
        raise ValueError("Expected two identifiers, 32 predictors, and the original label in prepared order.")

    model_columns = [*prepared_columns[2:-1], CLASS_COLUMN]
    validate_target((row.get(CLASS_COLUMN, "") for row in mapped_rows), allowed=CLASS_LABELS)
    model_rows = []
    for row in mapped_rows:
        validate_required_columns(row, model_columns)
        model_rows.append({column: row[column] for column in model_columns})
    return model_columns, model_rows


SPLIT_CONFIG_PATH = PROJECT_ROOT / "config" / "split.json"
SPLIT_ALGORITHM = "sha256_rank_stratified_two_stage_v1"
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_MANIFEST_COLUMNS = (*ID_COLUMNS, "split")
SPLIT_MANIFEST_PATH = PREPARATION_OUTPUT_DIR / "split_manifest.csv"


def _rounded_size(total: int, ratio: Fraction) -> int:
    """Compute the nearest integer, rounding half up with exact arithmetic."""
    return floor(total * ratio + Fraction(1, 2))


def _class_quotas(groups: Mapping[str, Sequence[str]], target_size: int) -> dict[str, int]:
    """Allocate an exact size by largest remainder, with original label order breaking ties."""
    total = sum(len(group) for group in groups.values())
    if not total or not 0 <= target_size <= total:
        raise ValueError("Invalid stratified allocation size.")

    quotas = {}
    remainders = {}
    for label in TARGET_LABELS:
        quotas[label], remainders[label] = divmod(target_size * len(groups[label]), total)
    remaining = target_size - sum(quotas.values())
    for label in sorted(TARGET_LABELS, key=remainders.get, reverse=True)[:remaining]:
        quotas[label] += 1
    return quotas


def _rank(seed: int, stage: str, encounter_id: str) -> tuple[bytes, int, str]:
    """Return the stable rank and a numeric ID tie-break with its original text."""
    digest = sha256(f"{seed}|{stage}|{encounter_id}".encode("ascii")).digest()
    return digest, int(encounter_id), encounter_id


def make_split_assignments(
    rows: Sequence[Mapping[str, str]], *, config_path: Path = SPLIT_CONFIG_PATH,
) -> dict[str, str]:
    """Allocate test first, then validation from development, by the original labels."""
    config = read_json(config_path)
    seed = config["seed"]
    if config["algorithm"] != SPLIT_ALGORITHM or config["stratify_by"] != TARGET_COLUMN:
        raise ValueError("Split configuration must use the two-stage rank and the original target.")
    if type(seed) is not int or seed < 0:
        raise ValueError("Split seed must be a nonnegative integer.")
    ratios = []
    for name in ("test_ratio", "validation_ratio_of_development"):
        value = config[name]
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string so allocation uses exact arithmetic.")
        ratio = Fraction(value)
        if not 0 < ratio < 1:
            raise ValueError(f"{name} must be between zero and one.")
        ratios.append(ratio)
    test_ratio, validation_ratio = ratios

    for column in ID_COLUMNS:
        validate_unique_ids((row.get(column, "") for row in rows), name=column)
    validate_target((row.get(TARGET_COLUMN, "") for row in rows), allowed=TARGET_LABELS)
    encounter_column = ID_COLUMNS[0]
    groups: dict[str, list[str]] = {label: [] for label in TARGET_LABELS}
    for row in rows:
        identifier = row[encounter_column]
        if not isinstance(identifier, str) or not identifier.isascii() or not identifier.isdecimal():
            raise ValueError("Encounter IDs must be strings of ASCII decimal digits.")
        groups[row[TARGET_COLUMN]].append(identifier)
    if any(not group for group in groups.values()):
        raise ValueError("All three original target labels must occur before splitting.")

    test_size = _rounded_size(len(rows), test_ratio)
    validation_size = _rounded_size(len(rows) - test_size, validation_ratio)
    assignments = {}
    for stage, size in (("test", test_size), ("validation", validation_size)):
        quotas = _class_quotas(groups, size)
        for label, identifiers in groups.items():
            ranked = sorted(identifiers, key=lambda identifier: _rank(seed, stage, identifier))
            quota = quotas[label]
            assignments.update(dict.fromkeys(ranked[:quota], stage))
            groups[label] = ranked[quota:]
    for identifiers in groups.values():
        assignments.update(dict.fromkeys(identifiers, "train"))
    return assignments


def _partition_statistics(
    source_rows: Sequence[Mapping[str, str]], columns: Sequence[str], model_rows: Sequence[Mapping[str, str]],
) -> dict:
    """Count original labels, binary classes, and empty fields for one export."""
    original_counts = Counter(row[TARGET_COLUMN] for row in source_rows)
    class_counts = Counter(row[CLASS_COLUMN] for row in model_rows)
    empty_counts = Counter()
    rows_with_empty_fields = 0
    for row in model_rows:
        empty_columns = [column for column, value in row.items() if value == ""]
        empty_counts.update(empty_columns)
        rows_with_empty_fields += bool(empty_columns)
    return {
        "rows": len(model_rows),
        "columns": len(columns),
        "original_labels": {label: original_counts[label] for label in TARGET_LABELS},
        "binary_classes": {label: class_counts[label] for label in CLASS_LABELS},
        "empty_cells_by_column": {column: empty_counts[column] for column in columns},
        "rows_with_empty_fields": rows_with_empty_fields,
    }


def save_model_partitions(
    prepared_columns: Sequence[str],
    mapped_rows: Sequence[Mapping[str, str]],
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, dict]:
    """Save and read back four model CSVs and the encounter-level split manifest."""
    assignments = make_split_assignments(
        mapped_rows, config_path=project_root / SPLIT_CONFIG_PATH.relative_to(PROJECT_ROOT),
    )
    mapping = load_target_mapping(project_root / TARGET_MAPPING_CONFIG_PATH.relative_to(PROJECT_ROOT))
    encounter_column, patient_column = ID_COLUMNS
    partitions: dict[str, list[Mapping[str, str]]] = {name: [] for name in SPLIT_NAMES}
    manifest_rows = []
    ordered = sorted(mapped_rows, key=lambda row: (int(row[encounter_column]), row[encounter_column]))
    for row in ordered:
        encounter, patient = row[encounter_column], row[patient_column]
        if row.get(CLASS_COLUMN) != mapping[row[TARGET_COLUMN]]:
            raise ValueError("Binary class disagrees with the original target mapping.")
        split = assignments[encounter]
        partitions[split].append(row)
        manifest_rows.append({encounter_column: encounter, patient_column: patient, "split": split})

    validate_partitions(
        ((row[encounter_column], row[patient_column]) for row in mapped_rows),
        {
            name: [(row[encounter_column], row[patient_column]) for row in records]
            for name, records in partitions.items()
        },
    )
    for name, records in partitions.items():
        if {row[TARGET_COLUMN] for row in records} != set(TARGET_LABELS):
            raise ValueError(f"At least one original target label is missing from {name}.")

    files = {}
    summaries = {}
    model_partitions = {}
    model_columns = []
    processed_dir = project_root / PROCESSED_DIR.relative_to(PROJECT_ROOT)
    for name, records in partitions.items():
        model_columns, model_rows = build_model_dataset(prepared_columns, records)
        model_partitions[name] = model_rows
        files[processed_dir / f"{name}.csv"] = (model_columns, model_rows)
        summaries[name] = _partition_statistics(records, model_columns, model_rows)

    combined = model_partitions["train"] + model_partitions["validation"]
    files[processed_dir / "train_validation.csv"] = (model_columns, combined)
    summaries["train_validation"] = _partition_statistics(
        partitions["train"] + partitions["validation"], model_columns, combined,
    )

    manifest_path = project_root / SPLIT_MANIFEST_PATH.relative_to(PROJECT_ROOT)
    files[manifest_path] = (SPLIT_MANIFEST_COLUMNS, manifest_rows)
    # Check existing outputs first, then write new files with the membership manifest last.
    for path in sorted(files, key=lambda path: not path.exists()):
        columns, records = files[path]
        write_csv(path, columns, records)
        _, reloaded = read_csv(path, expected_columns=columns)
        if reloaded != records:
            raise ValueError(f"CSV round trip changed partition data: {path}")
    return summaries


MODEL_ARFF_RELATION = "diabetes_readmission_32_features"
FULL_ARFF_RELATION = "diabetes_readmission_full_prepared"
ARFF_AUDIT_PATH = PREPARATION_OUTPUT_DIR / "arff_export_audit.json"
ARFF_MISSING_VALUES = frozenset(("", "?"))
_ARFF_ESCAPES = {
    "\\": "\\\\", "'": "\\'", '"': '\\"', "%": "\\%",
    "\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f",
}
_ARFF_UNESCAPES = {escaped[1]: value for value, escaped in _ARFF_ESCAPES.items()}
_ARFF_ESCAPE_PATTERN = re.compile(r"\\(.)")
_ARFF_NUMERIC = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_ARFF_FIELD = re.compile(r"(?:'(?P<nominal>(?:[^'\\\r\n]|\\.)*)'|(?P<plain>[^,'\r\n]+))(?P<separator>,|$)")


@dataclass(frozen=True)
class ArffAttribute:
    """An ordered ARFF attribute with a fixed nominal domain or a numeric type."""

    name: str
    domain: tuple[str, ...] | None = None


def create_arff_schema(
    predictors: Sequence[str],
    training_rows: Sequence[Mapping[str, str]],
    *,
    target: str,
    target_values: tuple[str, ...],
) -> tuple[ArffAttribute, ...]:
    """Fit predictor domains from the supplied rows and append the fixed target domain."""
    validate_schema((*predictors, target))
    if not training_rows:
        raise ValueError("ARFF schema requires at least one fitting row.")
    schema = []
    for name in predictors:
        if name in COUNT_COLUMNS:
            schema.append(ArffAttribute(name))
        else:
            values = {row[name] for row in training_rows}.difference(ARFF_MISSING_VALUES)
            if not values:
                raise ValueError(f"{name!r} has no observed values for its nominal ARFF domain.")
            schema.append(ArffAttribute(name, tuple(sorted(values))))
    schema.append(ArffAttribute(target, target_values))
    return tuple(schema)


def quote_arff(value: str) -> str:
    """Quote an ARFF literal and escape Weka's special characters."""
    if "\x00" in value:
        raise ValueError("ARFF literals cannot contain NUL.")
    return "'" + "".join(_ARFF_ESCAPES.get(char, char) for char in value) + "'"


def arff_header(relation: str, schema: Sequence[ArffAttribute]) -> str:
    """Serialize an ordered schema using Weka's dense ARFF syntax."""
    lines = [f"@RELATION {quote_arff(relation)}", ""]
    for attribute in schema:
        kind = "NUMERIC"
        if attribute.domain is not None:
            kind = "{" + ",".join(quote_arff(value) for value in attribute.domain) + "}"
        lines.append(f"@ATTRIBUTE {quote_arff(attribute.name)} {kind}")
    return "\n".join(lines) + "\n\n@DATA\n"


def make_arff(
    relation: str,
    schema: Sequence[ArffAttribute],
    rows: Sequence[Mapping[str, str]],
    *,
    allow_unseen: bool = False,
) -> tuple[str, dict]:
    """Serialize rows and distinguish source missing values from unseen categories."""
    columns = [attribute.name for attribute in schema]
    target = columns[-1]
    domains = {
        attribute.name: set(attribute.domain)
        for attribute in schema if attribute.domain is not None
    }
    nominal_tokens = {
        name: {value: quote_arff(value) for value in domain}
        for name, domain in domains.items()
    }
    original_missing: Counter[str] = Counter()
    unseen: Counter[str] = Counter()
    unseen_values: dict[str, Counter[str]] = {}
    with StringIO(newline="") as stream:
        stream.write(arff_header(relation, schema))
        for number, row in enumerate(rows, start=1):
            encoded = []
            for name in columns:
                value = row[name]
                if name == target and value not in domains[target]:
                    raise ValueError(f"Invalid ARFF target {value!r} in row {number}.")
                if value in ARFF_MISSING_VALUES:
                    original_missing[name] += 1
                    encoded.append("?")
                elif name not in domains:
                    if _ARFF_NUMERIC.fullmatch(value) is None or not isfinite(float(value)):
                        raise ValueError(f"Invalid numeric value for {name!r} in ARFF row {number}: {value!r}.")
                    encoded.append(value)
                elif value in domains[name]:
                    encoded.append(nominal_tokens[name][value])
                elif allow_unseen:
                    unseen[name] += 1
                    unseen_values.setdefault(name, Counter())[value] += 1
                    encoded.append("?")
                else:
                    raise ValueError(f"Unseen value for {name!r} in ARFF row {number}: {value!r}.")
            stream.write(",".join(encoded) + "\n")
        content = stream.getvalue()
    return content, {
        "rows": len(rows),
        "attributes": len(columns),
        "source_missing_by_column": dict(original_missing),
        "unseen_nominal_to_arff_missing": dict(unseen),
        "unseen_nominal_values_by_column": {name: dict(counts) for name, counts in unseen_values.items()},
        "arff_missing_by_column": dict(original_missing + unseen),
    }


def _read_arff_row(line: str) -> list[tuple[bool, str] | None]:
    """Decode one generated dense row, tracking quoted values and structural commas."""
    position = 0
    fields = []
    while position < len(line):
        field = _ARFF_FIELD.match(line, position)
        if field is None:
            raise ValueError("Invalid quoting or field syntax in generated ARFF.")
        nominal, plain, separator = field.groups()
        if nominal is not None:
            try:
                value = (True, nominal)
                if "\\" in nominal:
                    decoded = _ARFF_ESCAPE_PATTERN.sub(lambda match: _ARFF_UNESCAPES[match[1]], nominal)
                    value = (True, decoded)
            except KeyError as error:
                raise ValueError("Unknown escape in generated ARFF.") from error
        elif plain == "?":
            value = None
        elif _ARFF_NUMERIC.fullmatch(plain) is not None:
            value = (False, plain)
        else:
            raise ValueError("Expected a quoted nominal value, a number, or missing marker in ARFF.")
        fields.append(value)
        position = field.end()
        if separator == "," and position == len(line):
            raise ValueError("Trailing field separator in generated ARFF.")
    return fields


def validate_arff(
    content: str,
    relation: str,
    schema: Sequence[ArffAttribute],
    rows: Sequence[Mapping[str, str]],
    *,
    allow_unseen: bool = False,
) -> None:
    """Decode generated ARFF and check its header and values against source rows."""
    header = arff_header(relation, schema)
    if not content.startswith(header):
        raise ValueError("ARFF relation or ordered attribute schema differs.")
    domains = [None if attr.domain is None else set(attr.domain) for attr in schema]
    names = [attr.name for attr in schema]
    with StringIO(content, newline="\n") as stream:
        stream.seek(len(header))
        for number, (line, row) in enumerate(zip(stream, rows, strict=True), start=1):
            expected = []
            for name, domain in zip(names, domains, strict=True):
                value = row[name]
                if name == names[-1] and value not in domain:
                    raise ValueError(f"Invalid ARFF target in row {number}.")
                if value in ARFF_MISSING_VALUES or (allow_unseen and domain is not None and value not in domain):
                    expected.append(None)
                else:
                    if domain is not None and value not in domain:
                        raise ValueError(f"ARFF nominal value outside the declared domain in row {number}.")
                    expected.append((domain is not None, value))
            if _read_arff_row(line.removesuffix("\n")) != expected:
                raise ValueError(f"ARFF values differ from source in row {number}.")


def _verified_arff_source_splits(
    project_root: Path, prepared_rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    """Reconcile the saved membership manifest with the deterministic cohort split."""
    assignments = make_split_assignments(
        prepared_rows, config_path=project_root / SPLIT_CONFIG_PATH.relative_to(PROJECT_ROOT),
    )
    _, manifest = read_csv(
        project_root / SPLIT_MANIFEST_PATH.relative_to(PROJECT_ROOT),
        expected_columns=SPLIT_MANIFEST_COLUMNS,
    )
    encounter, patient = ID_COLUMNS
    partitions: dict[str, list[dict[str, str]]] = {name: [] for name in SPLIT_NAMES}
    ordered = sorted(prepared_rows, key=lambda row: (int(row[encounter]), row[encounter]))
    for row, membership in zip(ordered, manifest, strict=True):
        split = assignments[row[encounter]]
        if membership != {encounter: row[encounter], patient: row[patient], "split": split}:
            raise ValueError("Split manifest differs from the saved cohort or deterministic assignments.")
        partitions[split].append(row)
    return partitions


def save_weka_datasets(*, project_root: Path = PROJECT_ROOT) -> dict:
    """Write six validated ARFF datasets and their per-file conversion audit."""
    prepared_columns, prepared_rows = load_full_prepared(project_root=project_root)
    predictors = prepared_columns[2:-1]
    if not set(COUNT_COLUMNS).issubset(predictors):
        raise ValueError("The prepared data must include all configured numeric predictors.")
    partitions = _verified_arff_source_splits(project_root, prepared_rows)
    partitions["train_validation"] = partitions["train"] + partitions["validation"]
    model_columns = [*predictors, CLASS_COLUMN]
    mapping = load_target_mapping(project_root / TARGET_MAPPING_CONFIG_PATH.relative_to(PROJECT_ROOT))
    processed_dir = project_root / PROCESSED_DIR.relative_to(PROJECT_ROOT)
    weka_dir = project_root / WEKA_DIR.relative_to(PROJECT_ROOT)
    audit_path = project_root / ARFF_AUDIT_PATH.relative_to(PROJECT_ROOT)
    artifacts = {}
    audit = {"datasets": {}}

    full_schema = create_arff_schema(predictors, prepared_rows, target=TARGET_COLUMN, target_values=TARGET_LABELS)
    content, summary = make_arff(FULL_ARFF_RELATION, full_schema, prepared_rows)
    validate_arff(content, FULL_ARFF_RELATION, full_schema, prepared_rows)
    artifacts[weka_dir / "full_prepared.arff"] = content
    audit["datasets"]["full_prepared"] = {**summary, "schema_source": "full_prepared"}
    fitted_schemas = {}

    for name, schema_source in (
        ("train", "train"), ("validation", "train"), ("test", "train"),
        ("train_validation", "train_validation"), ("test_for_train_validation", "train_validation"),
    ):
        csv_name = "test" if name == "test_for_train_validation" else name
        _, rows = read_csv(processed_dir / f"{csv_name}.csv", expected_columns=model_columns)
        for number, (source, exported) in enumerate(zip(partitions[csv_name], rows, strict=True), start=1):
            if (
                exported[CLASS_COLUMN] != mapping[source[TARGET_COLUMN]]
                or any(exported[column] != source[column] for column in predictors)
            ):
                raise ValueError(f"{csv_name}.csv differs from the assigned cohort at row {number}.")

        if name == schema_source:
            fitted_schemas[name] = create_arff_schema(
                predictors, rows, target=CLASS_COLUMN, target_values=CLASS_LABELS,
            )
        schema = fitted_schemas[schema_source]
        allow_unseen = name != schema_source
        content, summary = make_arff(MODEL_ARFF_RELATION, schema, rows, allow_unseen=allow_unseen)
        validate_arff(content, MODEL_ARFF_RELATION, schema, rows, allow_unseen=allow_unseen)
        artifacts[weka_dir / f"{name}.arff"] = content
        audit["datasets"][name] = {**summary, "schema_source": schema_source}

    audit["domains_added_in_final_schema"] = {
        final.name: sorted(set(final.domain).difference(development.domain))
        for development, final in zip(fitted_schemas["train"], fitted_schemas["train_validation"], strict=True)
        if final.domain != development.domain
    }
    for path, content in artifacts.items():
        encoded = content.encode("utf-8")
        if path.exists() and path.read_bytes() != encoded:
            raise FileExistsError(f"Existing output differs: {path}")
        audit["datasets"][path.stem]["sha256"] = sha256(encoded).hexdigest()
    if audit_path.exists():
        write_json(audit_path, audit)

    for path, content in artifacts.items():
        write_text(path, content)
        if sha256_file(path) != audit["datasets"][path.stem]["sha256"]:
            raise ValueError(f"ARFF round trip changed file bytes: {path}")
    write_json(audit_path, audit)
    if read_json(audit_path) != audit:
        raise ValueError(f"ARFF audit did not survive the storage round trip: {audit_path}")
    return audit
