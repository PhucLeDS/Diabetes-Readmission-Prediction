"""Validate dataset structure and integrity without changing data."""

from collections.abc import Collection, Iterable, Mapping, Sequence


Identifier = str | int
Record = Mapping[str, object]
Membership = Iterable[tuple[Identifier, Identifier]]


def validate_required_columns(columns: Iterable[str], required: Iterable[str]) -> None:
    """Raise if any required column is absent."""
    missing = set(required).difference(columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}.")


def validate_schema(columns: Sequence[str], *, expected: Sequence[str] | None = None) -> None:
    """Require a nonempty header and optionally an exact column order."""
    if not columns or any(not column.strip() for column in columns):
        raise ValueError("A schema must contain nonempty column names.")
    if len(columns) != len(set(columns)):
        raise ValueError("A schema must contain unique column names.")
    if expected is not None and tuple(columns) != tuple(expected):
        raise ValueError(f"Unexpected columns or column order. Expected {list(expected)}, got {list(columns)}.")


def _column_value(row: Record, column: str, row_number: int) -> object:
    try:
        return row[column]
    except KeyError:
        raise ValueError(f"Data row {row_number} has no column {column!r}.") from None


def validate_types(
    rows: Iterable[Record],
    column_types: Mapping[str, type],
    *,
    nullable: Collection[str] = (),
) -> None:
    """Check exact Python types in parsed rows, allowing None in nullable columns."""
    for row_number, row in enumerate(rows, start=1):
        for column, expected_type in column_types.items():
            value = _column_value(row, column, row_number)
            if value is None and column in nullable:
                continue
            if type(value) is not expected_type:
                raise ValueError(
                    f"Data row {row_number}, column {column!r}: "
                    f"expected {expected_type.__name__}, got {type(value).__name__}."
                )


def validate_no_missing(
    rows: Iterable[Record],
    columns: Iterable[str],
    *,
    missing_values: Collection[object] = (None, ""),
) -> None:
    """Reject missing values in the specified columns using the caller's markers."""
    checked_columns = tuple(columns)
    for row_number, row in enumerate(rows, start=1):
        for column in checked_columns:
            if _column_value(row, column, row_number) in missing_values:
                raise ValueError(f"Missing value in data row {row_number}, column {column!r}.")


def _validate_identifier(value: Identifier, name: str) -> None:
    if type(value) not in (str, int) or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Invalid {name}: {value!r}.")


def validate_unique_ids(values: Iterable[Identifier], *, name: str = "ID") -> None:
    """Require valid, distinct identifiers within a single dataset."""
    seen = set()
    for value in values:
        _validate_identifier(value, name)
        if value in seen:
            raise ValueError(f"Duplicate {name}: {value!r}.")
        seen.add(value)


def validate_target(values: Iterable[str], *, allowed: Collection[str]) -> None:
    """Require every label to belong to the declared target domain."""
    for row_number, value in enumerate(values, start=1):
        if value not in allowed:
            raise ValueError(f"Unexpected target value in data row {row_number}: {value!r}.")


def validate_compatible_schemas(
    actual: Mapping[str, object], expected: Mapping[str, object],
) -> None:
    """Compare ordered attributes and their definitions, including types and domains."""
    validate_schema(tuple(actual), expected=tuple(expected))
    for column, definition in expected.items():
        if actual[column] != definition:
            raise ValueError(f"Incompatible attribute definition for {column!r}.")


def validate_partitions(source: Membership, partitions: Mapping[str, Membership]) -> None:
    """Check exact encounter coverage, patient pairing, and patient isolation."""
    if not partitions:
        raise ValueError("At least one partition is required.")

    expected_patients = {}
    for encounter, patient in source:
        _validate_identifier(encounter, "encounter ID")
        _validate_identifier(patient, "patient ID")
        if encounter in expected_patients:
            raise ValueError(f"Duplicate source encounter ID: {encounter!r}.")
        expected_patients[encounter] = patient

    assigned_encounters = set()
    patient_partitions = {}
    for name, membership in partitions.items():
        for encounter, patient in membership:
            _validate_identifier(encounter, "encounter ID")
            _validate_identifier(patient, "patient ID")
            if encounter not in expected_patients:
                raise ValueError(f"Unexpected encounter {encounter!r} in {name!r}.")
            if patient != expected_patients[encounter]:
                raise ValueError(f"Patient ID differs for encounter {encounter!r} in {name!r}.")
            if encounter in assigned_encounters:
                raise ValueError(f"Encounter {encounter!r} appears more than once in the partitions.")
            assigned_encounters.add(encounter)
            first_partition = patient_partitions.setdefault(patient, name)
            if first_partition != name:
                raise ValueError(f"Patient {patient!r} appears in both {first_partition!r} and {name!r}.")

    unassigned_count = len(expected_patients) - len(assigned_encounters)
    if unassigned_count:
        raise ValueError(f"{unassigned_count} source encounter(s) are missing from the partitions.")
