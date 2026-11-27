"""Classify laboratory result recording consistently across EDA tables."""

import pandas as pd


LAB_LEVELS = {
    "A1Cresult": ("Norm", ">7", ">8", "None"),
    "max_glu_serum": ("Norm", ">200", ">300", "None"),
}
LAB_COLUMNS = tuple(LAB_LEVELS)


def laboratory_result_recorded(values: pd.Series, column: str) -> pd.Series:
    """Return True for a valid result, False for literal None, and missing for unknowns."""
    return values.ne("None").astype("boolean").where(values.isin(LAB_LEVELS[column]))
