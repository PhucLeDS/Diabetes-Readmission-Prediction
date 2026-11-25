"""Read the verified prepared cohort for the Any Readmission analysis."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.io_utils import read_json
from src.paths import CONFIG_DIR, PROJECT_ROOT
from src.preparation.data_loading import COUNT_COLUMNS, ID_COLUMNS, TARGET_COLUMN
from src.preparation.dataset_export import (
    CLASS_COLUMN,
    TARGET_MAPPING_CONFIG_PATH,
    load_full_prepared,
    load_target_mapping,
)
from src.validation import validate_required_columns


ANALYSIS_CONFIG_PATH = CONFIG_DIR / "analysis.json"


@dataclass
class AnalysisInputs:
    """The predictors, binary outcome, and protocol for the full cohort."""

    predictors: pd.DataFrame
    target: pd.Series
    config: dict


def load_analysis_inputs(*, project_root: Path = PROJECT_ROOT) -> AnalysisInputs:
    """Read the prepared release and align its predictors with the mapped outcome."""
    config = read_json(project_root / ANALYSIS_CONFIG_PATH.relative_to(PROJECT_ROOT))
    if config["dataset"] != "full_prepared" or config["scope"] != "full_cohort_including_test":
        raise ValueError("Analysis requires the full prepared cohort, including the test patients.")

    columns, rows = load_full_prepared(project_root=project_root)
    predictor_columns = [column for column in columns if column not in (*ID_COLUMNS, TARGET_COLUMN)]
    questions = config["questions"]
    validate_required_columns(predictor_columns, (
        *questions["Q1"]["visit_columns"],
        questions["Q1"]["inpatient_column"],
        questions["Q2"]["stay_column"],
        questions["Q2"]["diagnoses_count_column"],
        questions["Q2"]["primary_diagnosis_column"],
        *questions["Q3"]["result_levels"],
        *questions["Q3"]["treatment_levels"],
    ))

    predictors = pd.DataFrame.from_records(rows, columns=predictor_columns).astype("string")
    for column in COUNT_COLUMNS:
        values = predictors[column]
        if not (values.eq("") | values.str.fullmatch(r"[0-9]+")).all():
            raise ValueError(f"{column!r} must contain nonnegative integers or empty fields.")
        predictors[column] = values.replace("", pd.NA).astype("Int64")

    mapping = load_target_mapping(project_root / TARGET_MAPPING_CONFIG_PATH.relative_to(PROJECT_ROOT))
    target = pd.Series(
        [mapping[row[TARGET_COLUMN]] for row in rows], index=predictors.index, name=CLASS_COLUMN, dtype="string",
    )
    return AnalysisInputs(predictors=predictors, target=target, config=config)
