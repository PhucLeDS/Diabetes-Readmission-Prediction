"""Export the full-cohort EDA tables, figures, and reproducibility metadata."""

from argparse import ArgumentParser
from dataclasses import fields
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from platform import python_version

import matplotlib
from matplotlib import ft2font
import numpy as np
import pandas as pd

from src.eda.admission_context import profile_admission_context
from src.eda.cohort_selection import MINIMUM_GROUP_SIZE, profile_cohort_selection
from src.eda.data import load_eda_inputs
from src.eda.descriptive import QUANTILES, profile_completeness, profile_numeric
from src.eda.diagnosis_detail import profile_diagnosis_detail
from src.eda.findings import summarize_findings
from src.eda.information_coverage import profile_information_coverage
from src.eda.nominal import RARE_LEVEL_PERCENT, profile_nominal
from src.eda.relationships import profile_relationships
from src.io_utils import input_checksums, write_artifacts
from src.paths import IDS_MAPPING_PATH, OUTPUT_DIR, PROJECT_ROOT
from src.preparation.data_cleaning import CLEANED_DATA_PATH
from src.preparation.dataset_export import TARGET_MAPPING_CONFIG_PATH
from src.preparation.dataset_preparation import (
    COHORT_DATA_PATH, FEATURE_AUDIT_PATH, FULL_PREPARED_PATH, ICD9_GROUP_RULE_VERSION,
    PREPARATION_CONFIG_PATH, PREPARATION_MANIFEST_PATH,
)


TABLE_DIR = Path("tables/eda")
FIGURE_DIR = Path("figures/eda")
MANIFEST_PATH = TABLE_DIR / "eda_manifest.json"
FIGURE_DPI = 150
INPUT_PATHS = {
    "prepared_cohort": FULL_PREPARED_PATH,
    "cleaned_encounters": CLEANED_DATA_PATH,
    "source_cohort": COHORT_DATA_PATH,
    "preparation_manifest": PREPARATION_MANIFEST_PATH,
    "preparation_config": PREPARATION_CONFIG_PATH,
    "feature_audit": FEATURE_AUDIT_PATH,
    "admission_lookup": IDS_MAPPING_PATH,
    "target_config": TARGET_MAPPING_CONFIG_PATH,
}
LF_INPUTS = ("preparation_config", "target_config")


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
                "discharge_eligibility_profile": "encounters in the eligibility group",
                "eligible_encounters_per_patient": "patients with at least one eligible encounter",
                "representative_encounter_profile": "encounters in the selection group among patients with multiple eligible encounters",
                "discharge_patient_overlap": "unique patients in the cleaned encounters",
                "lab_recording_by_prior_history": "patients in the prior history group",
                "lab_recording_by_characteristic": "patients at the reported characteristic level",
                "admission_type_source": "patients in the admission type",
                "age_by_admission": "patients with the reported admission code",
                "numeric_by_admission_zero_percent": "valid measurements for the reported admission code",
                "diagnosis_position_depth": "full cohort",
                "diagnosis_group_gain": "full cohort",
                "diagnosis_group_pairs": "full cohort",
                "diagnosis_source_codes": "patients with the group at any diagnosis position",
            },
            "selection_small_cell_count_exclusive": MINIMUM_GROUP_SIZE,
            "information_coverage": {
                "small_cell_count_exclusive": MINIMUM_GROUP_SIZE,
                "unknown_prior_history": "at least one prior visit count is missing",
                "unknown_lab_recording": "either laboratory value is empty or outside its source domain",
                "zero_denominator_percent": "empty CSV field",
            },
            "admission_context": {
                "small_cell_count_exclusive": MINIMUM_GROUP_SIZE,
                "numeric_small_valid_count_exclusive": MINIMUM_GROUP_SIZE,
                "zero_valid_percent": "empty CSV field",
            },
            "diagnosis_detail": {
                "grouping": ICD9_GROUP_RULE_VERSION,
                "coverage_groups": "mapped groups with Missing and Unmapped tracked separately",
                "pairs": "one count per patient per unordered pair of distinct mapped groups",
                "source_code_counts": "one count per patient per source code; empty codes excluded",
                "most_common_code_ties": "source code text order",
                "small_cell_count_exclusive": MINIMUM_GROUP_SIZE,
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


def export_eda(output_dir: Path = OUTPUT_DIR, *, overwrite: bool = False) -> Path:
    """Generate the same aggregate evidence that the notebook displays."""
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from src.eda.plots import (
        plot_admission_type_source, plot_diagnosis_group_gain, plot_information_coverage,
        plot_nominal_distributions, plot_numeric_distributions, plot_relationships,
    )

    inputs = input_checksums(INPUT_PATHS, root=PROJECT_ROOT, normalize_lf=LF_INPUTS)
    eda = load_eda_inputs()
    selection = profile_cohort_selection(eda)
    completeness = profile_completeness(eda.cohort, eda.admission_descriptions)
    coverage = profile_information_coverage(eda.cohort)
    numeric = profile_numeric(eda.cohort)
    nominal = profile_nominal(eda)
    admission = profile_admission_context(eda.cohort, nominal.admission_descriptions)
    diagnoses = profile_diagnosis_detail(eda.cohort, eda.source_cohort, nominal.diagnosis_summary)
    relationships = profile_relationships(eda.cohort)
    findings = summarize_findings(eda, completeness, numeric, nominal, relationships)
    tables = {
        "cohort_flow": eda.cohort_flow,
        "column_dictionary": eda.column_dictionary,
        "source_feature_decisions": eda.source_feature_decisions,
        "admission_lookup": eda.admission_descriptions,
        "eda_findings": findings,
    }
    for profile in (selection, completeness, coverage, numeric, nominal, admission, diagnoses, relationships):
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
            **plot_information_coverage(coverage, eda.admission_descriptions),
            **plot_nominal_distributions(nominal),
            "admission_type_source": plot_admission_type_source(admission.admission_type_source),
            "diagnosis_group_gain": plot_diagnosis_group_gain(diagnoses.diagnosis_group_gain),
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

    if inputs != input_checksums(INPUT_PATHS, root=PROJECT_ROOT, normalize_lf=LF_INPUTS):
        raise RuntimeError("An EDA input changed during export.")
    manifest = _eda_manifest(inputs, len(eda.cohort), eda.unclear_discharge_count, artifacts)
    artifacts[MANIFEST_PATH] = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    write_artifacts(output_dir, artifacts, overwrite=overwrite)
    return output_dir / MANIFEST_PATH


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Parent directory for tables and figures")
    parser.add_argument("--overwrite", action="store_true", help="Replace differing EDA artifacts")
    args = parser.parse_args()
    try:
        manifest = export_eda(args.output_dir, overwrite=args.overwrite)
    except FileExistsError as error:
        parser.exit(1, f"{error}. Use --overwrite or another --output-dir.\n")
    print(f"EDA manifest: {manifest}")


if __name__ == "__main__":
    main()
