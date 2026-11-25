# Data Preparation Report

This report documents preparation of the Any Readmission cohort and the verified CSV and Weka ARFF exports. The workflow retains 69,990 representative encounters from 101,766 source records. The model datasets use 32 baseline predictors and one binary class.

## 1. Data Loading

The loader verifies the SHA-256 checksum, column order, and row count of both source CSV files. It reads UTF-8 with optional BOM and preserves fields as strings, including identifiers, empty fields, and literal values such as `?` and `None`.

| File | Data rows | Columns | Size (bytes) |
| --- | ---: | ---: | ---: |
| `diabetic_data.csv` | 101,766 | 50 | 19,159,383 |
| `IDS_mapping.csv` | 67 | 2 | 2,547 |

Both SHA-256 checksums match the Phase 0 source audit.

- `diabetic_data.csv` — `0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97`
- `IDS_mapping.csv` — `f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e`

The mapping file is retained in its original form for reference, including separator rows and internal table headers.

### Encounter and Patient Counts

The encounter data contains **101,766 encounters** with unique `encounter_id` values and **71,518 patients**. Of these patients, **16,773** have multiple encounters. The loader checks both identifier columns and eight count columns as text representing nonnegative integers.

| Original `readmitted` label | Encounters | Share |
| --- | ---: | ---: |
| `NO` | 54,864 | 53.91% |
| `<30` | 11,357 | 11.16% |
| `>30` | 35,545 | 34.93% |

### Raw Markers

| Column | Original value | Cells |
| --- | --- | ---: |
| `race` | `?` | 2,273 |
| `weight` | `?` | 98,569 |
| `payer_code` | `?` | 40,256 |
| `medical_specialty` | `?` | 49,949 |
| `diag_1` | `?` | 21 |
| `diag_2` | `?` | 358 |
| `diag_3` | `?` | 1,423 |
| `gender` | `Unknown/Invalid` | 3 |
| `A1Cresult` | `None` | 84,748 |
| `max_glu_serum` | `None` | 96,420 |

The encounter CSV contains **192,849** cells with `?` and zero physically empty fields. Literal `None` values in the laboratory columns are counted separately. The notebook reproduces the profile for all 50 columns, patient counts, and original label distribution. [00_data_loading.ipynb](../notebooks/00_data_loading.ipynb)

## 2. Basic Data Cleaning

The cleaning step reads the verified encounter table and applies the rules in [preparation.json](../config/preparation.json). Seven columns use `?` as a missing marker, and `gender` uses `Unknown/Invalid`. The step writes these markers as empty CSV fields.

| Column | Source marker | Cells converted to empty |
| --- | --- | ---: |
| `race` | `?` | 2,273 |
| `weight` | `?` | 98,569 |
| `payer_code` | `?` | 40,256 |
| `medical_specialty` | `?` | 49,949 |
| `diag_1` | `?` | 21 |
| `diag_2` | `?` | 358 |
| `diag_3` | `?` | 1,423 |
| `gender` | `Unknown/Invalid` | 3 |
| **Total** | | **192,852** |

### Verification

| Measure | Before | After |
| --- | ---: | ---: |
| Encounters | 101,766 | 101,766 |
| Patients | 71,518 | 71,518 |
| Columns | 50 | 50 |
| Empty CSV fields | 0 | 192,852 |
| Encounters with at least one empty field | 0 | 100,723 |
| Literal `None` in `A1Cresult` | 84,748 | 84,748 |
| Literal `None` in `max_glu_serum` | 96,420 | 96,420 |

The cleaned CSV was read back and checked against the source cell by cell. The comparison verified each replacement and the preservation of every other value, including encounter IDs, patient IDs, original labels, and the order of records and columns. Both source checksums matched the values recorded in Data Loading.

The notebook produces `data/interim/cleaned.csv` and `outputs/preparation/cleaning_audit.csv`. The audit covers all 50 columns. Before cleaning, the reported missing rate counts the configured marker and existing empty fields. After cleaning, it counts empty fields. [01_data_preparation.ipynb](../notebooks/01_data_preparation.ipynb)

## 3. Cohort Selection

The selection step uses the cleaned encounter data and the discharge codes in [preparation.json](../config/preparation.json). It first excludes death and hospice encounters, then selects the eligible encounter with the smallest numeric `encounter_id` for each `patient_nbr`. The original ID breaks ties when two strings represent the same number. The output uses the same ordering rule. The source data has no encounter timestamps for establishing chronological order.

| Stage | Encounters | Unique patients | Removed at this step | `NO` | `<30` | `>30` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cleaned | 101,766 | 71,518 | 0 | 54,864 | 11,357 | 35,545 |
| Eligible | 99,343 | 69,990 | 2,423 | 52,527 | 11,314 | 35,502 |
| Cohort | 69,990 | 69,990 | 29,353 | 41,479 | 6,285 | 22,226 |

| Excluded discharge code | Encounters removed |
| --- | ---: |
| `11` | 1,642 |
| `13` | 399 |
| `14` | 372 |
| `19` | 8 |
| `20` | 2 |
| `21` | 0 |

The final cohort retains 2,474 encounters with discharge code `18` (`NULL`) and 778 with code `25` (`Not Mapped`). These codes are outside the exclusion list.

### Column Audit

| Column | Empty in cleaned data | Empty in eligible records | Empty in cohort |
| --- | ---: | ---: | ---: |
| `race` | 2,273 | 2,234 | 1,919 |
| `gender` | 3 | 3 | 3 |
| `weight` | 98,569 | 96,218 | 67,202 |
| `payer_code` | 40,256 | 39,398 | 30,416 |
| `medical_specialty` | 49,949 | 48,616 | 33,654 |
| `diag_1` | 21 | 20 | 10 |
| `diag_2` | 358 | 356 | 293 |
| `diag_3` | 1,423 | 1,419 | 1,224 |
| **Total empty fields** | **192,852** | **188,264** | **134,721** |

The audit records the number of empty fields and the missing rate with each stage's row count as the denominator for all 50 columns. The other 42 columns have zero empty fields in all three stages.

The saved `data/interim/cohort.csv` was read back and compared with an independent selection from the cleaned CSV. All 69,990 records matched across 50 columns. Each selected encounter met the discharge policy, and each encounter ID and patient ID appears once. The notebook also writes the column audit to `outputs/preparation/cohort_audit.csv`. [01_data_preparation.ipynb](../notebooks/01_data_preparation.ipynb)

## 4. Feature Transformation

The fixed ICD-9 grouping rule `icd9_fixed_v1` replaces `diag_1`, `diag_2`, and `diag_3` with `diag_1_group`, `diag_2_group`, and `diag_3_group`. Numeric codes are grouped by the integer portion. The parser trims surrounding whitespace and normalizes letter case for nonempty values. The groups describe broad categories for analysis. The parser checks code format and applies the grouping rules.

| Diagnosis group | Rule | `diag_1_group` | `diag_2_group` | `diag_3_group` |
| --- | --- | ---: | ---: | ---: |
| `Diabetes` | 250, including decimal subcodes | 5,748 | 9,700 | 12,547 |
| `Circulatory` | 390–459 or 785 | 21,390 | 22,084 | 20,867 |
| `Respiratory` | 460–519 or 786 | 9,491 | 6,928 | 4,652 |
| `Digestive` | 520–579 or 787 | 6,488 | 2,856 | 2,700 |
| `Genitourinary` | 580–629 or 788 | 3,441 | 5,330 | 4,049 |
| `Musculoskeletal` | 710–739 | 4,064 | 1,295 | 1,368 |
| `Injury` | 800–999 | 4,696 | 1,824 | 1,410 |
| `Neoplasms` | 140–239 | 2,538 | 1,600 | 1,146 |
| `Other_numeric` | Other well-formed numeric codes | 11,205 | 16,293 | 16,510 |
| `Other_V` | Well-formed V-prefixed codes | 918 | 1,217 | 2,586 |
| `Other_E` | Well-formed E-prefixed codes | 1 | 570 | 931 |
| `Missing` | Empty source field | 10 | 293 | 1,224 |
| `Unmapped` | Value outside the accepted formats | 0 | 0 | 0 |
| **Total** | | **69,990** | **69,990** | **69,990** |

Before grouping, the three source columns contained 694, 723, and 756 distinct nonempty codes, respectively. Each grouped column has 11 observed code groups and the `Missing` category. Its `Missing` count matches the corresponding source column's empty-field count.

The notebook produced 69,990 rows and 50 columns after grouping. Comparison with `cohort.csv` verified the other 47 fields on every row, including both IDs and the original target. The notebook holds the transformed table in memory for predictor selection. [01_data_preparation.ipynb](../notebooks/01_data_preparation.ipynb)

## 5. Feature Selection

The selection step applies the ordered baseline in [preparation.json](../config/preparation.json). It keeps `encounter_id` and `patient_nbr` first, the 32 predictors in the configured order, and the original `readmitted` target last. The predictors include 8 numeric encounter and utilization counts and 24 nominal attributes covering demographics, admission details, laboratory results, medications, and grouped diagnoses.

| Decision | Columns | Reason |
| --- | --- | --- |
| Drop source columns | `weight`, `payer_code`, `medical_specialty` | The reference training set showed 96.05%, 43.59%, and 47.96% missing, respectively. |
| Drop source columns | `examide`, `citoglipton`, `troglitazone`, `glimepiride-pioglitazone` | Each contained only `No` in the reference training set. |
| Drop source columns | `acetohexamide`, `tolbutamide`, `miglitol`, `tolazamide`, `glipizide-metformin`, `metformin-rosiglitazone`, `metformin-pioglitazone` | Each had between 1 and 18 values other than `No` in the reference training set. |
| Drop source column | `discharge_disposition_id` | Used to determine cohort eligibility. Its availability at the intended prediction point remains unverified. |
| Replace source diagnoses during grouping | `diag_1`, `diag_2`, `diag_3` | The baseline retains their grouped counterparts. |
| Omit derived columns | `total_prior_visits`, `has_prior_inpatient`, `A1C_tested`, `glucose_tested`, `admission_type_desc`, `admission_source_desc`, `discharge_disposition_desc` | The pipeline uses the original visit counts, laboratory categories, and admission or discharge codes at their relevant steps. |

The 15 source columns are dropped from the already grouped schema. The audit also records the 3 earlier diagnosis replacements and the 7 derived columns that the pipeline omits. An audit of the 44,794 encounters assigned to Train reproduced the baseline rationale. Empty-field counts were 43,024 for `weight`, 19,527 for `payer_code`, and 21,483 for `medical_specialty`. The four constant medication columns and the seven counts of values other than `No` also matched. The three source diagnosis columns had 645, 672, and 700 distinct nonempty values in Train. Predictive value remains an empirical question.

### Verified Output

| Measure | Grouped cohort | Selected table |
| --- | ---: | ---: |
| Encounters | 69,990 | 69,990 |
| Columns | 50 | 35 |
| Empty fields | 133,194 | 1,922 |

All retained values were checked against the grouped cohort. The original labels remain `NO` (41,479), `<30` (6,285), and `>30` (22,226). The selected table contains 1,919 empty `race` fields and 3 empty `gender` fields across 1,920 encounters. The reduction in empty fields comes from column removal.

The notebook writes `outputs/preparation/feature_audit.csv` with 60 entries. It records output positions, column roles, and before-and-after missingness where the column exists. The selected table is held in memory for the next step, which writes the full prepared CSV. [01_data_preparation.ipynb](../notebooks/01_data_preparation.ipynb)

## 6. Full Prepared Dataset

[01_data_preparation.ipynb](../notebooks/01_data_preparation.ipynb) builds `data/processed/full_prepared.csv` from the verified raw files. Its column order is `encounter_id`, `patient_nbr`, the 32 configured predictors, and the original `readmitted` target.

| Measure | Verified value |
| --- | ---: |
| Encounters | 69,990 |
| Unique patients | 69,990 |
| Columns | 35 |
| Original labels `NO` | 41,479 |
| Original labels `<30` | 6,285 |
| Original labels `>30` | 22,226 |
| Empty `race` fields | 1,919 |
| Empty `gender` fields | 3 |
| Encounters with at least one empty field | 1,920 |

The export verified the ordered schema, unique IDs, and every prepared value against the cohort. Reading the CSV back reproduced the in-memory table. The two raw SHA-256 checksums matched before and after export.

The CSV has SHA-256 `0207daf2cb4f83700cf71337d39519aef7c7683e910a7ba6511216e6601a787f`. The local `outputs/preparation/preparation_manifest.json` records the schema, row and label counts, per-column empty counts, cohort flow, configuration checksum, and the checksums of 6 dataset and audit files.

Both preparation notebooks ran from fresh kernels. A second run of the preparation notebook from raw files in a separate directory reproduced all 6 files and the manifest byte for byte.

## 7. Target Transformation

The binary target follows the reference project's Any Readmission mapping, as applied by `load_label_config()` and `load_labels()` in its `src/weka/weka_export.py`. The repository stores the mapping in [target_mapping.json](../config/target_mapping.json). The positive class, `readmitted`, includes both recorded timing categories.

| Original label | Binary class | Encounters in the saved cohort |
| --- | --- | ---: |
| `NO` | `not_readmitted` | 41,479 |
| `<30` | `readmitted` | 6,285 |
| `>30` | `readmitted` | 22,226 |
| **Total** | | **69,990** |

| Binary class | Encounters | Share of cohort |
| --- | ---: | ---: |
| `not_readmitted` | 41,479 | 59.26% |
| `readmitted` | 28,511 | 40.74% |

[02_split_export.ipynb](../notebooks/02_split_export.ipynb) reads the saved CSV, checks its checksum, schema, original label counts, and identifier uniqueness, then constructs the binary target in memory. The loader accepts LF or CRLF representations of the preparation configuration when the checksum for one representation matches the manifest. The data CSV still requires an exact byte match.

The internal table has 36 columns, including both identifiers and both target representations. The model view has 33 columns, consisting of the 32 ordered predictors and `class` in the final position. Every predictor value was verified against the saved CSV across all 69,990 rows. The model view retains 1,919 empty `race` fields and 3 empty `gender` fields across 1,920 rows. The source CSV retains its original three labels and the checksum recorded in Section 6. The binary counts also match the totals across the reference Any Readmission partitions.

## 8. Data Splitting

The split reads the saved `full_prepared.csv` and uses the policy in [split.json](../config/split.json). The reference implementation is `make_split_assignments()`, `_rank()`, `_one_fifth_rounded()`, and `_class_quotas()` in `src/preparation/data_splitting.py` of the reference project.

1. Group the cohort by the three original `readmitted` labels. Set the test size to 20% of the cohort, rounded to the nearest integer with half ties rounded up.
2. Allocate that size across labels by largest remainder. Assign the integer portion of each proportional quota first, then distribute remaining slots by fractional remainder, breaking ties in the order `NO`, `<30`, `>30`.
3. Within each label, rank encounters by the SHA-256 digest of the ASCII string `42|test|encounter_id`, where `encounter_id` is the ID value. Break hash ties by numeric encounter ID, then by its original text. Assign the lowest ranks in each class to test.
4. Apply the same size, quota, and ranking rules to select 20% of the remaining development encounters for validation, using the stage name `validation` in the hash. Assign the remaining encounters to train.

The primary CSVs preserve ascending numeric encounter order, with original ID text breaking ties. The combined CSV appends validation rows after train rows in their respective orders. Each model file uses 32 predictors followed by the binary `class` in a 33-column UTF-8 CSV. The split manifest stores `encounter_id`, `patient_nbr`, and `split` in encounter order.

| Dataset | Rows | Original `NO` | Original `<30` | Original `>30` | `not_readmitted` | `readmitted` | Readmitted share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 44,794 | 26,547 | 4,022 | 14,225 | 26,547 | 18,247 | 40.74% |
| Validation | 11,198 | 6,636 | 1,006 | 3,556 | 6,636 | 4,562 | 40.74% |
| Test | 13,998 | 8,296 | 1,257 | 4,445 | 8,296 | 5,702 | 40.73% |
| Train + validation | 55,992 | 33,183 | 5,028 | 17,781 | 33,183 | 22,809 | 40.74% |

| Dataset | Empty `race` fields | Empty `gender` fields | Rows with at least one empty field |
| --- | ---: | ---: | ---: |
| Train | 1,207 | 2 | 1,208 |
| Validation | 315 | 0 | 315 |
| Test | 397 | 1 | 397 |
| Train + validation | 1,522 | 2 | 1,523 |

All other exported columns have zero empty fields. CSV read-back confirmed ordered schema and field values, including empty fields and literal laboratory categories.

The manifest covers all 69,990 encounters with no repeated encounters or patients across the primary splits. Every assignment and all four model CSVs match the reference membership, values, and row order. Reversing input row order produced the same assignments. A separate run from the saved cohort reproduced the four model CSVs and the split manifest byte for byte. The full prepared CSV and its preparation manifest retained their checksums.

[02_split_export.ipynb](../notebooks/02_split_export.ipynb) presents the verified file counts and missingness. The combined file is intended for final training once model selection is complete, and test remains reserved for final evaluation.

## 9. Dataset Export

The Weka export reads the five saved CSVs and reconciles their row order, predictor values, and labels with the prepared cohort and split manifest. The model serialization follows `create_schema()`, `quote_arff()`, `arff_header()`, and `make_arff()` in the reference project's `src/weka/weka_export.py`. The final training schema follows `build_final_outputs()` in `src/weka/export_train_validation.py`. This implementation adds a full-cohort exploration ARFF, reconciles the source CSVs before writing, and records the exact categories converted to missing.

Each ARFF has 33 ordered attributes with 8 numeric predictors and 24 nominal predictors before the final outcome attribute. Admission codes remain nominal. `full_prepared.arff` uses relation `diabetes_readmission_full_prepared` and the original `readmitted` domain `{NO,<30,>30}`. The five modeling files use relation `diabetes_readmission_32_features` with the Any Readmission domain `{not_readmitted,readmitted}` for `class`. In Weka, select `class` as the final model attribute and refer to the positive class by name, `readmitted`.

| File | Rows | Predictor domains fitted on | Missing source fields | Unseen values converted to `?` |
| --- | ---: | --- | ---: | ---: |
| `full_prepared.arff` | 69,990 | Full prepared cohort | 1,922 | 0 |
| `train.arff` | 44,794 | Train | 1,209 | 0 |
| `validation.arff` | 11,198 | Train | 315 | 2 |
| `test.arff` | 13,998 | Train | 398 | 2 |
| `train_validation.arff` | 55,992 | Train + Validation | 1,524 | 0 |
| `test_for_train_validation.arff` | 13,998 | Train + Validation | 398 | 2 |

Empty fields and source missing markers are serialized as `?`. Literal laboratory `None` and diagnosis group `Missing` are quoted nominal values. In Validation, both unseen values were `admission_source_id = 14`. That code becomes part of the Train + Validation schema. Both Test files retain the exact same 13,998 encounters in the same order. Their data sections are identical, with one `admission_source_id = 11` field and one `chlorpropamide = Down` field converted to `?` in each file. Their only header difference is the inclusion of admission-source code `14` in the final training schema. The local `outputs/preparation/arff_export_audit.json` records the conversion counts, added domain values, and output checksums.

Weka 3.9.7 loaded all six files. A Java check using Weka's `Instances` API verified relation names, ordered attribute types and domains, row counts, and every parsed value against the CSVs and the applicable nominal policy. Weka also confirmed header compatibility within each modeling stage and correctly decoded a separate fixture with quotes, backslashes, punctuation, control characters, and UTF-8 text. The five model ARFFs preserve the reference attribute declarations and data records.

[02_split_export.ipynb](../notebooks/02_split_export.ipynb) ran from a fresh kernel. A separate run from the saved cohort reproduced the four model CSVs, split manifest, six ARFFs, and ARFF audit byte for byte. Use `full_prepared.arff` for exploration, `test.arff` with the Train schema, and `test_for_train_validation.arff` with the Train + Validation schema.

## 10. Final Validation

The end-to-end suite ran all three preparation notebooks in fresh kernels using Python 3.13.5. Each of two independent workspaces started with the two official raw files, notebook sources, code, and configuration. The run generated 19 files. Their SHA-256 checksums were identical across both workspaces, and both raw files matched their approved checksums after processing.

| Check | Verified result |
| --- | --- |
| Raw integrity and cleaning | Compared 101,766 encounters across 50 columns. Only the 192,852 approved marker cells changed. The cleaning audit reconciled per column, and laboratory `None` values were preserved. |
| Cohort, diagnosis, and predictor schema | Rebuilt the lowest eligible encounter per patient independently from the cleaned CSV. All 69,990 cohort rows and ordered prepared values matched. The three diagnosis groups, numeric predictor types, original target counts, and missing-field counts reconciled with the saved data. |
| Target and partitions | Verified the binary target on every model row and the original label counts per split. All 69,990 encounters and patients appeared exactly once across Train, Validation, and Test. Train + Validation matched the exact ordered concatenation of 55,992 rows. |
| CSV and ARFF consistency | Checked all six ARFF headers, row counts, and decoded values against their CSV sources and fitting scopes. Verified the unseen nominal values and the output checksums in the ARFF audit. |
| Reproducibility | Both clean runs reproduced all intermediate data, final datasets, manifests, and audits byte for byte. The rerun in the working repository after the ARFF rename also preserved the checksums of the existing inputs and generated outputs. |

The full suite passed 61 tests, including the six end-to-end checks. The documented command-line workflow also generated all 19 files in a clean workspace and preserved the notebook source files. Run it from the repository root with the project environment active using `python -m unittest discover -s tests -v`. If either official raw CSV is absent, the integration class reports a skip and the unit tests can still run. The Weka runtime check is recorded in Section 9. The byte-for-byte benchmark uses the same configuration bytes and Python version for both runs. The preparation manifest records the configuration's byte checksum and the Python version. The files generated from the current checkout matched the saved release artifacts. The preparation manifest recorded a different configuration hash because the checkout uses CRLF and the earlier run used LF. Both hashes resolve to the same configuration content under the loader's line-ending check.

See the [end-to-end tests](../tests/preparation/test_preparation_pipeline.py) for the checks applied to the generated files.

## 11. Methodological Limitations

- **Cohort selection.** The smallest eligible numeric encounter ID serves as a reproducible selection rule. The input has no encounter timestamps to establish chronological order. The cohort applies the documented death and hospice exclusions and retains 3,252 records with discharge codes `18` and `25`, whose descriptions leave disposition unclear.
- **Target coverage.** The two recorded readmission categories form a single positive class. The source records category labels and supplies no exact days to readmission. The scope of readmissions outside the source observation coverage remains unverified.
- **Predictor availability and representation.** The intended prediction point is near discharge. Availability of each attribute at that point requires independent clinical or operational verification. Broad ICD-9 groups discard code detail, and `None` in the laboratory columns leaves it unclear whether a test was performed. The fixed baseline still requires evaluation.
- **Generalizability and evaluation.** The splits partition the historical cohort by patient and preserve original outcome proportions approximately. They do not establish performance on later cohorts, held-out hospitals, or external populations. Predictive performance, subgroup performance, and clinical applicability remain unevaluated.
