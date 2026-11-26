"""Export the cohort Analysis tables, figures, and reproducibility manifest."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from platform import python_version

import matplotlib
from matplotlib import ft2font
import numpy as np
import pandas as pd

from src.analysis.clinical_records import profile_clinical_records
from src.analysis.current_encounters import profile_current_encounters
from src.analysis.data import ANALYSIS_CONFIG_PATH, AnalysisInputs, load_analysis_inputs
from src.analysis.evaluation import build_evaluation_map
from src.analysis.prior_visits import profile_prior_visits
from src.analysis.rule_comparisons import compare_rule_conditions
from src.io_utils import input_checksums, write_artifacts
from src.paths import OUTPUT_DIR, PROJECT_ROOT
from src.preparation.dataset_export import CLASS_LABELS, TARGET_MAPPING_CONFIG_PATH
from src.preparation.dataset_preparation import (
    FULL_PREPARED_PATH,
    PREPARATION_CONFIG_PATH,
    PREPARATION_MANIFEST_PATH,
)


TABLE_DIR = Path("tables/analysis")
FIGURE_DIR = Path("figures/analysis")
MANIFEST_PATH = TABLE_DIR / "analysis_manifest.json"
FIGURE_DPI = 150
INPUT_PATHS = {
    "analysis_config": ANALYSIS_CONFIG_PATH,
    "prepared_cohort": FULL_PREPARED_PATH,
    "preparation_manifest": PREPARATION_MANIFEST_PATH,
    "preparation_config": PREPARATION_CONFIG_PATH,
    "target_config": TARGET_MAPPING_CONFIG_PATH,
}
LF_INPUTS = ("analysis_config", "preparation_config", "target_config")


def _analysis_tables(analysis: AnalysisInputs) -> dict[str, pd.DataFrame]:
    prior = profile_prior_visits(analysis)
    encounters = profile_current_encounters(analysis)
    clinical = profile_clinical_records(analysis)
    rules = compare_rule_conditions(analysis)
    return {
        "prior_visit_summary": prior.prior_visit_summary,
        "prior_visit_combinations": prior.prior_visit_combinations,
        "prior_inpatient_levels": prior.prior_inpatient_levels,
        "history_rule_confusion_matrix": prior.history_rule_confusion_matrix,
        "stay_by_history": encounters.stay_by_history,
        "diagnoses_count_by_history": encounters.diagnoses_count_by_history,
        "primary_diagnosis_by_history": encounters.primary_diagnosis_by_history,
        "lab_recording_by_outcome": clinical.lab_recording_by_outcome,
        "lab_results_by_outcome": clinical.lab_results_by_outcome,
        "treatment_by_outcome": clinical.treatment_by_outcome,
        "rule_comparisons": rules,
        "analysis_evaluation_map": build_evaluation_map(analysis, prior, encounters, clinical, rules),
    }


def _figure_artifacts(tables: dict[str, pd.DataFrame]) -> dict[Path, bytes]:
    from matplotlib import pyplot as plt

    from src.analysis.plots import plot_lab_recording_by_outcome, plot_prior_visit_summary, plot_stay_by_history

    plotters = {
        "prior_visit_summary": plot_prior_visit_summary,
        "stay_by_history": plot_stay_by_history,
        "lab_recording_by_outcome": plot_lab_recording_by_outcome,
    }
    artifacts = {}
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        for name, plot in plotters.items():
            figure = plot(tables[name])
            try:
                with BytesIO() as buffer:
                    figure.savefig(buffer, dpi=FIGURE_DPI, format="png")
                    artifacts[FIGURE_DIR / f"{name}.png"] = buffer.getvalue()
            finally:
                plt.close(figure)
    return artifacts


def _analysis_manifest(analysis: AnalysisInputs, inputs: dict, artifacts: dict[Path, bytes]) -> dict:
    return {
        "manifest_version": 1,
        "scope": {
            "dataset": analysis.config["dataset"],
            "unit": analysis.config["unit"],
            "prediction_point": analysis.config["prediction_point"],
            "includes_test_patients": True,
            "patients": len(analysis.target),
            "predictors": analysis.predictors.shape[1],
            "outcome_counts": {label: int(analysis.target.eq(label).sum()) for label in CLASS_LABELS},
        },
        "configuration": analysis.config,
        "inputs": inputs,
        "runtime": {
            "python": python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "freetype": ft2font.__freetype_version__,
        },
        "statistics": {
            "positive_class": CLASS_LABELS[1],
            "denominators": {
                "coverage_percent": "All cohort patients",
                "readmission_rate_percent": "Patients in the group",
                "share_of_readmissions_percent": "All cohort patients with recorded readmission",
                "precision_percent": "Patients alerted by the rule among complete histories",
                "recall_percent": "Patients with recorded readmission among complete histories",
                "percent_of_recorded_results": "Patients with a recorded result for the corresponding test",
                "support_percent": "All cohort patients",
                "confidence_percent": "Patients with a confirmed antecedent match",
                "lift": "Cohort readmission rate, with rule confidence as the numerator",
            },
            "zero_denominator": "Undefined",
            "confidence_difference_units": "Percentage points, calculated before display rounding",
            "csv": {
                "encoding": "UTF-8",
                "line_endings": "LF",
                "index": False,
                "undefined": "Empty field",
                "laboratory_no_result": analysis.config["questions"]["Q3"]["no_recorded_result"],
                "float_rounding": "Display only",
            },
            "figures": {"format": "PNG", "dpi": FIGURE_DPI, "backend": "Agg"},
        },
        "limitations": {
            "cohort": "The smallest eligible encounter ID selects a representative encounter. Chronology is unavailable.",
            "evaluation": "The Analysis includes Test patients. Record any influence on later development decisions.",
            "predictor_availability": "Availability of encounter features at the intended prediction point is unverified.",
        },
        "artifact_sha256": {
            path.as_posix(): sha256(content).hexdigest() for path, content in artifacts.items()
        },
    }


def export_analysis(output_dir: Path = OUTPUT_DIR, *, overwrite: bool = False) -> Path:
    """Export the notebook's aggregate results using the verified cohort."""
    matplotlib.use("Agg")
    inputs = input_checksums(INPUT_PATHS, root=PROJECT_ROOT, normalize_lf=LF_INPUTS)
    analysis = load_analysis_inputs()
    tables = _analysis_tables(analysis)
    artifacts = {
        TABLE_DIR / f"{name}.csv": table.to_csv(index=False, lineterminator="\n").encode("utf-8")
        for name, table in tables.items()
    }
    artifacts.update(_figure_artifacts(tables))
    if inputs != input_checksums(INPUT_PATHS, root=PROJECT_ROOT, normalize_lf=LF_INPUTS):
        raise RuntimeError("An Analysis input changed during export.")

    manifest = _analysis_manifest(analysis, inputs, artifacts)
    artifacts[MANIFEST_PATH] = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    write_artifacts(output_dir, artifacts, overwrite=overwrite)
    return output_dir / MANIFEST_PATH


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Parent directory for tables and figures")
    parser.add_argument("--overwrite", action="store_true", help="Replace differing Analysis artifacts")
    args = parser.parse_args()
    try:
        manifest = export_analysis(args.output_dir, overwrite=args.overwrite)
    except FileExistsError as error:
        parser.exit(1, f"{error}. Use --overwrite or another --output-dir.\n")
    print(f"Analysis manifest: {manifest}")


if __name__ == "__main__":
    main()
