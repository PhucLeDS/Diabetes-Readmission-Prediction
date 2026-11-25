"""Read, write, and fingerprint project data files."""

import csv
import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from filecmp import cmp
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TextIO

from src.validation import validate_schema


CsvValue = str | int | float | None


def ensure_directory(path: Path) -> Path:
    """Create a directory and any missing parents."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of the exact file bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(
    path: Path, *, expected_columns: Sequence[str] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    """Read a UTF-8 CSV table as text, preserving empty fields and literal tokens."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        columns = next(reader, [])
        validate_schema(columns, expected=expected_columns)
        rows = []
        for row_number, values in enumerate(reader, start=1):
            if len(values) != len(columns):
                raise ValueError(f"{path}: unexpected field count in data row {row_number}.")
            rows.append(dict(zip(columns, values, strict=True)))
    return columns, rows


class _LfCsvStream:
    """Keep CSV quoting for CR and LF while writing LF record separators."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream

    def write(self, record: str) -> int:
        return self.stream.write(record.removesuffix("\r\n") + "\n")


@contextmanager
def _temporary_output(path: Path, *, overwrite: bool) -> Iterator[Path]:
    ensure_directory(path.parent)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        yield temporary_path

        if overwrite:
            temporary_path.replace(path)
        else:
            try:
                if os.name == "nt":
                    # Windows rename fails when the destination already exists.
                    temporary_path.rename(path)
                else:
                    path.hardlink_to(temporary_path)
            except FileExistsError:
                if not cmp(temporary_path, path, shallow=False):
                    raise FileExistsError(f"Existing output differs: {path}") from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@contextmanager
def _output_stream(path: Path, *, overwrite: bool) -> Iterator[TextIO]:
    with _temporary_output(path, overwrite=overwrite) as temporary_path:
        with temporary_path.open("w", encoding="utf-8", newline="") as stream:
            yield stream


def write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, CsvValue]],
    *,
    overwrite: bool = False,
) -> Path:
    """Write an ordered UTF-8 CSV table, using empty fields for Python None."""
    header = tuple(columns)
    validate_schema(header)
    expected_keys = set(header)
    with _output_stream(path, overwrite=overwrite) as stream:
        writer = csv.writer(_LfCsvStream(stream))
        writer.writerow(header)
        for row_number, row in enumerate(rows, start=1):
            if row.keys() != expected_keys:
                raise ValueError(f"{path}: columns differ from the schema in data row {row_number}.")
            writer.writerow(row[column] for column in header)
    return path


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON configuration or manifest."""
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, data: object, *, overwrite: bool = False) -> Path:
    """Write a deterministic UTF-8 JSON configuration or manifest."""
    with _output_stream(path, overwrite=overwrite) as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def write_text(path: Path, data: str, *, overwrite: bool = False) -> Path:
    """Write UTF-8 text with the supplied line endings and protected overwrite behavior."""
    with _output_stream(path, overwrite=overwrite) as stream:
        stream.write(data)
    return path


def write_bytes(path: Path, data: bytes, *, overwrite: bool = False) -> Path:
    """Publish bytes using the same overwrite protection as the text writers."""
    with _temporary_output(path, overwrite=overwrite) as temporary_path:
        temporary_path.write_bytes(data)
    return path
