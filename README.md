# Diabetes Readmission Prediction

This project studies hospital readmission among patients with diabetes using data mining and machine learning. The workflow uses Python, Jupyter Notebook, and Weka to prepare data, conduct analyses, and develop models. The [data preparation report](reports/01_data_preparation.md) documents verified results from source checks through model CSV export.

## Contents

- [Research Objective](#research-objective)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Analysis and Modeling](#analysis-and-modeling)
- [Repository Structure](#repository-structure)
- [Installation and Usage](#installation-and-usage)
- [Reproducibility and Validation](#reproducibility-and-validation)

## Research Objective

Predict whether a patient has a recorded hospital readmission following an eligible encounter. The intended prediction point is near discharge.

The unit of analysis is one representative encounter per patient. The selected predictors cover demographics, admission information, prior healthcare utilization, laboratory results, medications, and diagnoses.

The binary target is derived from the original `readmitted` attribute.

| Original label | Model class | Meaning |
| --- | --- | --- |
| `NO` | `not_readmitted` | No readmission recorded |
| `<30` | `readmitted` | Readmission recorded |
| `>30` | `readmitted` | Readmission recorded |

The positive class is `readmitted`. The `not_readmitted` label reflects the scope of observation in the dataset.

## Dataset

The UCI [Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008) dataset describes inpatient encounters across 130 US hospitals and healthcare networks. It is published under the CC BY 4.0 license.

| File | Content |
| --- | --- |
| `diabetic_data.csv` | 101,766 encounters, 71,518 patients, and 50 columns |
| `IDS_mapping.csv` | Descriptions of admission type, admission source, and discharge disposition codes |

Source files are stored in `data/raw/`. The literal `None` in `A1Cresult` and `max_glu_serum` is preserved as a categorical value reflecting recorded result information. This value alone does not establish whether a test was performed.

## Data Preparation

The preparation workflow validates source data, normalizes missing values, selects the cohort, groups ICD-9 diagnosis codes, and applies the selected set of 32 predictors. Audits record row counts, column counts, and missing values before and after transformations.

Cohort selection excludes encounters with the specified discharge codes related to death or hospice. For each patient, it then selects the eligible encounter with the smallest numeric `encounter_id`. This ID identifies the representative record. The source data does not contain timestamps that establish chronological encounter order.

The full prepared dataset retains two identifiers, 32 predictors, and the original `readmitted` label. The predictor set contains 8 numeric and 24 nominal attributes. It is a selected baseline that will be assessed through model experiments. The exported CSV was verified on 69,990 encounters and contains 1,922 empty fields across `race` and `gender`.

The full dataset is stratified by the three original labels using seed `42` and deterministic SHA-256 ranking. The split assigns approximately 64% to train, 16% to validation, and 20% to test. Each model dataset contains 32 predictors and `class`. The combined file appends validation to train for final model fitting.

The data-loading and cleaning steps each verified 101,766 encounters and 50 columns. Cleaning converted 192,852 source markers to empty fields. Cohort and split counts were verified against the saved CSVs. All 69,990 split assignments match the reference run.

| Dataset stage | Rows |
| --- | ---: |
| Raw data | 101,766 |
| After excluding ineligible encounters | 99,343 |
| Cohort with one encounter per patient | 69,990 |
| Train | 44,794 |
| Validation | 11,198 |
| Test | 13,998 |
| Train + validation | 55,992 |

In the cohort, 41,479 encounters are `not_readmitted` and 28,511 are `readmitted`. The positive class represents 40.74% of 69,990 encounters.

### Data Products

| File or directory | Purpose |
| --- | --- |
| `data/interim/cleaned.csv` | Data after cleaning |
| `data/interim/cohort.csv` | Cohort with one representative encounter per patient |
| `data/processed/full_prepared.csv` | Full dataset with 35 columns, including identifiers and the original label |
| `data/processed/train.csv`, `validation.csv`, `test.csv` | Model datasets in CSV format |
| `data/processed/train_validation.csv` | Combined train and validation dataset |
| `data/weka/full_prepared.arff` | 32 predictors and the original label for exploration in Weka |
| `data/weka/train.arff`, `validation.arff`, `test.arff` | Model datasets using the train schema |
| `data/weka/train_validation.arff`, `test_for_train_validation.arff` | Model datasets using the train + validation schema |
| `outputs/preparation/` | Audits, split manifest, and preparation manifest |

Nominal domains for model ARFF files are built from train during development and from train + validation for final training. Unseen nominal values are represented as `?` and recorded in the audit. Both test files use the same membership with their respective schemas.

## Analysis and Modeling

**EDA and Analysis** examine data quality, missing values, attribute distributions, and relationships with the target. Analyses that inform predictor or preprocessing decisions use the training set.

Selected EDA findings were verified on the reference training set of 44,794 encounters.

- `weight` has 43,024 missing values (96.05%), `medical_specialty` has 21,483 (47.96%), and `payer_code` has 19,527 (43.59%). These columns are excluded from the baseline.
- Four medication columns, `examide`, `citoglipton`, `troglitazone`, and `glimepiride-pioglitazone`, contain only `No`. Seven other medication columns have between 1 and 18 values other than `No`. These columns are excluded from the baseline.
- The three original diagnosis columns contain 645, 672, and 700 distinct codes on train, respectively. Grouping reduces the number of categories and may remove detail.

These are reference findings. EDA and Analysis on data generated by this repository require execution and verification.

**Data Mining** is pending. The methods and results will be added after the research approach is selected.

**Model Training and Evaluation** are pending. During method comparison, data-dependent transformations are fit on train or on the corresponding training fold. Validation is used to select an approach. After the configuration is finalized, the pipeline may be fit again on train + validation and evaluated on test.

## Repository Structure

| Directory | Purpose |
| --- | --- |
| `data/` | Raw, interim, processed, and Weka data |
| `src/` | Data processing, validation, and analysis logic |
| `notebooks/` | Workflow execution, results, and commentary |
| `config/` | Processing and experiment configurations |
| `outputs/` | Audits and generated results |
| `reports/` | Methods and results |
| `tests/` | Unit and integration tests |
| `weka_java/` | Java code for data mining and model training |

This structure describes the components of the full project. Processing logic resides in `src/`. Notebooks call functions, present results, and may call functions that save finalized datasets.

## Installation and Usage

The initial environment was verified with Python 3.13.5. On Windows, run the following commands from the repository root using Git Bash.

```bash
py -3.13 -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

Download `diabetic_data.csv` and `IDS_mapping.csv` from the UCI dataset page and place them in [data/raw/](data/raw/). Direct Python dependencies are listed in [requirements.txt](requirements.txt).

Run [00_data_loading.ipynb](notebooks/00_data_loading.ipynb) from a fresh kernel to verify the source checksums and inspect the schema, patient coverage, and original labels. Run [01_data_preparation.ipynb](notebooks/01_data_preparation.ipynb) to normalize the approved markers, select the cohort, and write both datasets and their column audits. The notebook then groups the three ICD-9 fields, applies the ordered predictor list, and writes `full_prepared.csv` with its feature audit and preparation manifest.

Run [02_split_export.ipynb](notebooks/02_split_export.ipynb) to verify the saved cohort, apply the Any Readmission mapping, and write the four model CSVs and the split manifest. The notebook displays class distributions and empty-field counts from the verified exports.

According to [.gitignore](.gitignore), data files and files in `outputs/preparation/` remain local. When cloning the repository, prepare the source data and run the workflow to regenerate the outputs.

## Reproducibility and Validation

Configurations, checksums, schemas, and audits support reconciliation from raw data to exported datasets. Validation covers label values, row counts, identifier uniqueness, patient isolation, CSV/ARFF compatibility, and split reproducibility. Validation results and methods are recorded in [reports/](reports/).

The availability of each predictor near discharge has not been independently verified. The cohort retains 3,252 encounters with discharge descriptions recorded as `NULL` or `Not Mapped`. The historical dataset, missing-value recording practices, and diagnosis grouping require consideration when interpreting results. Clinical applicability and generalizability remain unverified.
