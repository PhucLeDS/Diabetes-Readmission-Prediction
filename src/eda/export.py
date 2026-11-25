"""Export the full-cohort EDA tables, figures, and reproducibility metadata."""

from argparse import ArgumentParser
from dataclasses import fields
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from platform import python_version
from tempfile import NamedTemporaryFile

import matplotlib
from matplotlib import ft2font
import numpy as np
import pandas as pd

from src.eda.data import load_eda_inputs
from src.eda.descriptive import QUANTILES, profile_completeness, profile_numeric
from src.eda.findings import summarize_findings
from src.eda.nominal import RARE_LEVEL_PERCENT, profile_nominal
from src.eda.relationships import profile_relationships
from src.io_utils import sha256_file
from src.paths import IDS_MAPPING_PATH, OUTPUT_DIR, PROJECT_ROOT
from src.preparation.dataset_export import TARGET_MAPPING_CONFIG_PATH
from src.preparation.dataset_preparation import (
    FEATURE_AUDIT_PATH, FULL_PREPARED_PATH, PREPARATION_CONFIG_PATH, PREPARATION_MANIFEST_PATH,
)


TABLE_DIR = Path("tables/eda")
FIGURE_DIR = Path("figures/eda")
MANIFEST_PATH = TABLE_DIR / "eda_manifest.json"
FIGURE_DPI = 150
INPUT_PATHS = {
    "prepared_cohort": FULL_PREPARED_PATH,
    "preparation_manifest": PREPARATION_MANIFEST_PATH,
    "preparation_config": PREPARATION_CONFIG_PATH,
    "feature_audit": FEATURE_AUDIT_PATH,
    "admission_lookup": IDS_MAPPING_PATH,
    "target_config": TARGET_MAPPING_CONFIG_PATH,
}


def _input_checksums() -> dict:
    inputs = {}
    for name, path in INPUT_PATHS.items():
        normalize = name in {"preparation_config", "target_config"}
        digest = sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if normalize else sha256_file(path)
        inputs[name] = {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": digest,
            "checksum_mode": "LF line endings" if normalize else "exact file bytes",
        }
    return inputs


def _eda_manifest(inputs: dict, cohort_size: int, unclear_discharge_count: int, artifacts: dict[Path, bytes]) -> dict:
    return {
        "manifest_version": 1,
        "scope": {
            "dataset": "full_prepared",
            "unit": "one representative encounter per patient",
            "patients": cohort_size,
            "unclear_discharge_count": unclear_discharge_count,
        },
        "inputs": inputs,
        "runtime": {
            "python": python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "freetype": ft2font.__freetype_version__,
        },
        "statistics": {
            "quantile_interpolation": "linear",
            "numeric_quantiles": QUANTILES,
            "percent_formula": "100 * count / denominator",
            "denominators": {
                "default": "full cohort",
                "numeric_zero_percent": "valid observations in the numeric column",
                "laboratory_result_share": "patients with a result recorded for the corresponding test",
                "age_prior_visit_share": "patients in the age band",
                "medication_flag_share": "patients in the source flag group",
            },
            "rare_level_percent_exclusive": RARE_LEVEL_PERCENT,
            "empty_csv_fields_are_missing": True,
            "retained_source_categories": ["None", "Missing", "Unmapped"],
            "csv": {"encoding": "UTF-8", "line_endings": "LF", "float_rounding": "display only"},
            "figures": {"format": "PNG", "dpi": FIGURE_DPI, "backend": "Agg"},
        },
        "artifact_sha256": {
            path.as_posix(): sha256(content).hexdigest() for path, content in artifacts.items()
        },
    }


def _write_artifacts(output_dir: Path, artifacts: dict[Path, bytes], *, overwrite: bool) -> None:
    for relative, content in artifacts.items():
        path = output_dir / relative
        if path.exists() and path.read_bytes() != content and not overwrite:
            raise FileExistsError(f"Existing EDA output differs: {path}. Use --overwrite or another --output-dir.")
    for relative, content in artifacts.items():
        path = output_dir / relative
        if path.exists() and path.read_bytes() == content:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # Publish a complete file from the destination filesystem.
        temp = None
        try:
            with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
                temp = Path(stream.name)
                stream.write(content)
            if overwrite:
                temp.replace(path)
            else:
                try:
                    if os.name == "nt":
                        temp.rename(path)
                    else:
                        path.hardlink_to(temp)
                except FileExistsError:
                    if path.read_bytes() != content:
                        raise
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)


def export_eda(output_dir: Path = OUTPUT_DIR, *, overwrite: bool = False) -> Path:
    """Generate the same aggregate evidence that the notebook displays."""
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from src.eda.plots import plot_nominal_distributions, plot_numeric_distributions, plot_relationships

    inputs = _input_checksums()
    eda = load_eda_inputs()
    completeness = profile_completeness(eda.cohort, eda.admission_descriptions)
    numeric = profile_numeric(eda.cohort)
    nominal = profile_nominal(eda)
    relationships = profile_relationships(eda.cohort)
    findings = summarize_findings(eda, completeness, numeric, nominal, relationships)
    tables = {
        "cohort_flow": eda.cohort_flow,
        "column_dictionary": eda.column_dictionary,
        "source_feature_decisions": eda.source_feature_decisions,
        "admission_lookup": eda.admission_descriptions,
        "eda_findings": findings,
    }
    for profile in (completeness, numeric, nominal, relationships):
        for field in fields(profile):
            value = getattr(profile, field.name)
            if isinstance(value, pd.DataFrame):
                tables[field.name] = value

    artifacts = {
        TABLE_DIR / f"{name}.csv": table.to_csv(index=False, lineterminator="\n").encode("utf-8")
        for name, table in tables.items()
    }
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        figures = {
            **plot_numeric_distributions(numeric.numeric_frequency, cohort_size=len(eda.cohort)),
            **plot_nominal_distributions(nominal),
            **plot_relationships(relationships),
        }
        try:
            for name, figure in figures.items():
                with BytesIO() as buffer:
                    figure.savefig(buffer, dpi=FIGURE_DPI, format="png")
                    artifacts[FIGURE_DIR / f"{name}.png"] = buffer.getvalue()
        finally:
            for figure in figures.values():
                plt.close(figure)

    if inputs != _input_checksums():
        raise RuntimeError("An EDA input changed during export.")
    manifest = _eda_manifest(inputs, len(eda.cohort), eda.unclear_discharge_count, artifacts)
    artifacts[MANIFEST_PATH] = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_artifacts(output_dir, artifacts, overwrite=overwrite)
    return output_dir / MANIFEST_PATH


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Parent directory for tables and figures")
    parser.add_argument("--overwrite", action="store_true", help="Replace differing EDA artifacts")
    args = parser.parse_args()
    try:
        manifest = export_eda(args.output_dir, overwrite=args.overwrite)
    except FileExistsError as error:
        parser.exit(1, f"{error}\n")
    print(f"EDA manifest: {manifest}")


if __name__ == "__main__":
    main()
