# Data Preparation Report

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
