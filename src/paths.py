"""Canonical paths for the diabetes readmission project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
WEKA_DIR = DATA_DIR / "weka"
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PREPARATION_OUTPUT_DIR = OUTPUT_DIR / "preparation"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

RAW_DATA_PATH = RAW_DIR / "diabetic_data.csv"
IDS_MAPPING_PATH = RAW_DIR / "IDS_mapping.csv"
