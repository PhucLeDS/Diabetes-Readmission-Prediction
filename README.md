# Diabetes Readmission Prediction

This project uses the Diabetes 130-US Hospitals dataset to study whether a patient has a recorded hospital readmission after an eligible inpatient encounter. The workflow uses Python and Jupyter for data preparation, exploration, and analysis, and Weka for data mining and model evaluation. [Data Preparation Report](reports/01_data_preparation.md) · [EDA Report](reports/02_eda.md) · [Analysis Report](reports/03_analysis.md)

## Contents

- [Research Objective](#research-objective)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Readmission Analysis](#readmission-analysis)
- [Data Mining and Modeling](#data-mining-and-modeling)
- [Repository Structure](#repository-structure)
- [Installation and Usage](#installation-and-usage)
- [Reproducibility and Validation](#reproducibility-and-validation)
- [Limitations](#limitations)

## Research Objective

Predict whether a patient has a recorded readmission following an eligible inpatient encounter. The intended prediction point is near discharge. The analysis uses one representative encounter per patient.

| Original `readmitted` value | Model class | Meaning |
| --- | --- | --- |
| `NO` | `not_readmitted` | No readmission recorded |
| `<30` | `readmitted` | Readmission recorded |
| `>30` | `readmitted` | Readmission recorded |

The positive class is `readmitted`. A negative label reflects the dataset's observation coverage.

## Dataset

The source contains historical inpatient encounters from 130 US hospitals and healthcare networks from 1999 through 2008. It is available under CC BY 4.0. [UCI Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008)

| Source file | Contents |
| --- | --- |
| `diabetic_data.csv` | 101,766 encounters, 71,518 patients, and 50 columns |
| `IDS_mapping.csv` | Descriptions of admission type, admission source, and discharge disposition codes |

The loader verifies each source file by SHA-256. Laboratory `None` values remain nominal categories. The [source audit](reports/01_data_preparation.md#1-data-loading) records the exact file sizes, checksums, and raw value counts.

## Data Preparation

The pipeline validates the source, normalizes approved missing markers, and excludes encounters with the specified death and hospice discharge codes. It selects the eligible encounter with the smallest numeric `encounter_id` for each patient. It then groups three ICD-9 fields and applies the configured baseline of 32 predictors, consisting of 8 numeric and 24 nominal attributes. The saved cohort contains 69,990 encounters and 1,922 empty predictor fields across `race` and `gender`. [Preparation policy](config/preparation.json)

| Dataset | Encounters | Recorded readmissions |
| --- | ---: | ---: |
| Raw | 101,766 | 46,902 |
| Eligible | 99,343 | 46,816 |
| Prepared cohort | 69,990 | 28,511 |
| Train | 44,794 | 18,247 |
| Validation | 11,198 | 4,562 |
| Test | 13,998 | 5,702 |
| Train + Validation | 55,992 | 22,809 |

Train, Validation, and Test account for approximately 64%, 16%, and 20% of the cohort. The split uses seed `42`, deterministic SHA-256 ranking, and stratification by the three original outcome labels. The combined training file appends Validation to Train. [Split policy](config/split.json)

### Data Products

| Files | Contents |
| --- | --- |
| `data/interim/cleaned.csv`, `data/interim/cohort.csv` | Cleaned encounters and the selected patient cohort |
| `data/processed/full_prepared.csv` | 35 columns with two IDs, 32 predictors, and the original outcome |
| `data/processed/train.csv`, `validation.csv`, `test.csv`, `train_validation.csv` | Four CSVs with 32 predictors and the binary `class` |
| `data/weka/full_prepared.arff` | Cohort exploration file with the original outcome |
| `data/weka/train.arff`, `validation.arff`, `test.arff` | Development datasets sharing nominal domains learned from Train |
| `data/weka/train_validation.arff`, `test_for_train_validation.arff` | Final training and holdout datasets sharing domains learned from Train + Validation |
| `outputs/preparation/` | Cleaning, cohort, feature, and ARFF audits plus the preparation and split manifests |

Both Test ARFFs represent the same held-out encounters. The exporter uses `?` for missing values and unseen nominal predictor values, whose counts and tokens are recorded in the ARFF audit.

## Exploratory Data Analysis

The [EDA notebook](notebooks/03_eda.ipynb) describes the full cohort of 69,990 patients, including all 8 numeric predictors, 24 nominal predictors, and the Any Readmission outcome distribution. The prepared data contains 1,922 empty fields across 1,920 patients. A separate laboratory summary shows that 54,041 patients have neither test result recorded. The [EDA Report](reports/02_eda.md) explains the denominators, evidence, and limits of these summaries.

EDA uses the entire prepared cohort, including patients assigned to Test. Record any influence of full-cohort exploration on later modeling decisions when interpreting the final evaluation.

## Readmission Analysis

The Analysis notebook uses the full cohort of **69,990 patients** to examine how prior service use, encounter information, and combined conditions relate to Any Readmission. A rule that alerts on at least one recorded prior service visit identifies **9,849/28,511 positive cases (34.54%)** and misses **18,662 cases**. The analysis also compares length of stay, laboratory result coverage, and the confidence of two combined conditions. The generated `analysis_evaluation_map.csv` links each finding to Weka comparisons, metrics, and evaluation data. [Analysis notebook](notebooks/04_analysis.ipynb) · [Analysis Report](reports/03_analysis.md)

The descriptive analysis includes patients assigned to Test. Record any influence of the full-cohort results on development choices and disclose it when interpreting the final evaluation.

## Data Mining and Modeling

Develop rules and models on Train. Fit data-dependent transforms on Train or within each training fold, and use Validation to compare approaches. Once the approach is selected, fit again on Train + Validation and evaluate on the matching Test export. Data mining and model evaluation are the next stages in this repository.

## Repository Structure

| Directory | Purpose |
| --- | --- |
| `data/` | Source files, intermediate data, and generated CSV and ARFF datasets |
| `src/` | Data preparation, EDA, Analysis, I/O, and validation logic |
| `notebooks/` | Workflow entry points and displayed results |
| `config/` | Cleaning, cohort, predictor, target, split, and Analysis policies |
| `outputs/preparation/` | Generated audits and manifests |
| `outputs/tables/eda/` | Locally generated aggregate EDA tables and manifest |
| `outputs/figures/eda/` | Locally generated EDA figures |
| `outputs/tables/analysis/` | Locally generated Analysis tables, Weka evaluation reference, and manifest |
| `outputs/figures/analysis/` | Locally generated Analysis figures |
| `reports/` | Methods and verified preparation, EDA, and Analysis results |
| `tests/` | Unit and end-to-end tests |
| `weka_java/` | Reserved directories for data mining and model training code |

## Installation and Usage

### Set up Python

Use Python 3.13. The workflow was verified on Python 3.13.5. On Windows, run these commands from the repository root in Git Bash.

```bash
py -3.13 -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name diabetes-readmission \
  --display-name "Python (Diabetes Readmission)"
```

The pinned packages provide the notebook environment and pandas for data preparation, EDA, and Analysis. Preparation uses pandas for table processing and CSV export. EDA and Analysis also use the declared libraries for statistics and figures. [Python requirements](requirements.txt)

### Place the raw inputs

Download the original `diabetic_data.csv` and `IDS_mapping.csv` from the UCI dataset and place both CSVs directly in [data/raw/](data/raw/). Keep the source bytes intact so the checksum verification succeeds.

### Generate the datasets

Run the notebooks in order. In a Jupyter-compatible editor, select the **Python (Diabetes Readmission)** kernel and run every cell in each notebook.

| Notebook | Result |
| --- | --- |
| [00_data_loading.ipynb](notebooks/00_data_loading.ipynb) | Verify the sources and display their structure, missing markers, and outcome counts |
| [01_data_preparation.ipynb](notebooks/01_data_preparation.ipynb) | Write the intermediate data, full prepared CSV, and preparation audits |
| [02_split_export.ipynb](notebooks/02_split_export.ipynb) | Write four model CSVs, six ARFFs, the split manifest, and the ARFF audit |

The same workflow runs from Git Bash with the environment active.

```bash
jupyter execute --kernel_name=diabetes-readmission --timeout=180 \
  notebooks/00_data_loading.ipynb \
  notebooks/01_data_preparation.ipynb \
  notebooks/02_split_export.ipynb
```

This command executes the notebooks without saving cell outputs to their source files. Data and preparation artifacts are local. The writers protect existing artifacts from conflicting content. If an older artifact blocks a run, archive the affected generated files and their audit or manifest before rerunning the corresponding notebooks. Keep both raw inputs in place. [Git ignore rules](.gitignore)

### Explore the Prepared Cohort

After generating `full_prepared.csv` and the preparation audits, open [03_eda.ipynb](notebooks/03_eda.ipynb), select the **Python (Diabetes Readmission)** kernel, and run every cell. From the repository root with the project environment active, generate the local tables, figures, and manifest:

```bash
python -m src.eda.export
```

Use `--output-dir` to export to a separate location. The exporter reuses identical existing files and stops if an existing artifact differs. Use `--overwrite` to replace differing local EDA artifacts after reviewing the change. Generated EDA tables and figures are ignored by Git and can be recreated with this command.

### Analyze Readmission

After generating `full_prepared.csv` and the preparation manifest, open [04_analysis.ipynb](notebooks/04_analysis.ipynb), select the **Python (Diabetes Readmission)** kernel, and run every cell. From the repository root with the environment active, generate the Analysis outputs with the following command.

```bash
python -m src.analysis.export
```

The command creates **12 CSV tables, 3 PNG figures, and one manifest** in the Analysis output directories. These files are generated locally and ignored by Git. Use `--output-dir` for a separate destination and `--overwrite` to replace files with different contents after reviewing the change.

### Use the ARFF files in Weka

Open the relevant ARFF in Weka Explorer under **Preprocess**. For a modeling file, set the last attribute, `class`, as the class and identify the positive value by name, `readmitted`. Pair Train with the development schema exports and Train + Validation with `test_for_train_validation.arff`. Successful loading and header compatibility were verified in Weka 3.9.7. [Dataset export checks](reports/01_data_preparation.md#9-dataset-export)

## Reproducibility and Validation

Run the suite from the repository root with the project environment active.

```bash
python -m unittest discover -s tests -v
```

After the pandas refactor, all **85 tests** passed, comprising 67 Preparation tests, 3 EDA tests, and 15 Analysis tests. They cover source integrity, attribute order, partition membership, CSV and ARFF values, checksums, reconciliation across tables, and output protection. Integration tests require the corresponding input data. [Data Preparation validation](reports/01_data_preparation.md#10-final-validation) · [EDA validation](reports/02_eda.md#9-reproduction-and-validation) · [Analysis validation](reports/03_analysis.md#8-reproduction-and-validation)

Two independent runs executed all five notebooks in fresh kernels and exported the EDA and Analysis artifacts. Each run reproduced **19 Preparation files, 35 EDA files, and 16 Analysis files** byte for byte against the pre-refactor baseline and the other run. Both raw files retained their approved checksums. Weka **3.9.7** loaded all six ARFF files and verified compatible headers for each training stage. [Data Preparation validation](reports/01_data_preparation.md#10-final-validation)

## Limitations

The encounter ID supports deterministic selection. Establishing chronology, confirming predictor availability at discharge, and verifying outcomes outside the dataset's observation coverage require additional evidence. The cohort retains 3,252 encounters with unclear discharge codes. The current holdout comes from the historical source cohort. Temporal, hospital-level, and external validation remain pending. [Methodological limitations](reports/01_data_preparation.md#11-methodological-limitations)
