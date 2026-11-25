# Exploratory Data Analysis Report

## 1. Purpose and Scope

This report describes the prepared dataset for the Any Readmission prediction task. It covers cohort selection, data completeness, feature distributions, and selected relationships between attributes.

The primary input is `data/processed/full_prepared.csv`, which contains **69,990 patients** and one representative inpatient encounter per patient. The table has **35 columns**: two identifiers for reconciliation, **32 predictors** comprising 8 numeric and 24 nominal attributes, and the original `readmitted` label. The generated `column_dictionary.csv` describes each field's role, source, unit or domain, and period.

The EDA is descriptive and uses the entire cohort, including patients assigned to the test set. Outcome labels are summarized at cohort level. Feature, preprocessing, and model decisions should use the development data. Document any influence that full-cohort exploration has on those decisions when interpreting the final test evaluation. The selection and preparation rules are detailed in the [Data Preparation Report](01_data_preparation.md).

## 2. Cohort Selection

| Stage | Encounters | Unique patients | Encounters removed at this stage |
| --- | ---: | ---: | ---: |
| Cleaned, with the same rows as the raw source | 101,766 | 71,518 | 0 |
| Eligible after discharge-code filtering | 99,343 | 69,990 | 2,423 |
| One representative encounter per patient | 69,990 | 69,990 | 29,353 |

The pipeline excludes encounters with the configured death or hospice discharge codes, then selects the eligible encounter with the smallest numeric ID for each patient. This rule is reproducible. The source has no encounter timestamps to establish whether the selected encounter was the patient's first admission. The cohort retains **3,252 encounters** whose discharge descriptions are unclear. See `cohort_flow.csv` and `eda_manifest.json`.

## 3. Statistical Conventions

- **Numeric predictors.** The profile reports valid and missing counts, zeros, minimum, Q1, median, Q3, P95, P99, and maximum. Quantiles use linear interpolation over valid observations. The zero percentage uses the valid count for that column. The frequency table retains every observed value.
- **Nominal predictors.** Frequencies and percentages use the full cohort. A rare level is an observed nonempty category present in less than 1% of the cohort. This flag describes frequency. Source codes, rare values, and empty fields remain visible in the tables.
- **Group percentages.** Laboratory coverage uses the full cohort, while each recorded result distribution uses patients with a result for that specific test. Age comparisons use the number of patients in each age band. Medication-flag comparisons report both the share within each source-flag group and the share of the full cohort.

## 4. Completeness and Source Categories

The prepared CSV contains **1,922 empty fields across 1,920 patients (2.74%)**. There are 1,919 empty `race` fields and 3 empty `gender` fields. Two patients are missing both.

The diagnosis groups use `Missing` for an empty source code and `Unmapped` for a code outside the grouping rules. The counts of `Missing` at the three diagnosis positions are **10, 293, and 1,224**, respectively. No `Unmapped` values appear in the cohort.

`admission_type_id` and `admission_source_id` use separate lookup tables. Every observed code has a lookup entry. Descriptions are unclear for **7,893/69,990 (11.28%)** of admission types and **5,069/69,990 (7.24%)** of admission sources. These are column-specific rates, and a patient can appear in both groups. See `missing_summary.csv`, `missing_patterns.csv`, `special_values.csv`, and `admission_descriptions.csv`.

### Laboratory Results

| Test | Patients with a recorded result | Share of 69,990 patients | Most frequent recorded level |
| --- | ---: | ---: | --- |
| HbA1c | 12,846 | 18.35% | `>8` in 6,239/12,846 patients (48.57%) |
| Serum glucose | 3,349 | 4.78% | `Norm` in 1,701/3,349 patients (50.79%) |

A total of **246** patients have both results recorded, **12,600** have only HbA1c, **3,103** have only serum glucose, and **54,041 (77.21%)** have neither result recorded. Neither laboratory column contains an empty CSV field. The literal `None` is retained as a separate category for records without a result. The source does not establish whether a test was performed for each such record. See `lab_recording_patterns.csv`, `laboratory_results.csv`, and `laboratory_result_distributions.png`.

## 5. Cohort Characteristics

### Current Encounter and Prior Service Use

The median length of stay is **3 days**, with Q1 at 2 days and Q3 at 6 days. The median numbers of laboratory procedures, non-laboratory procedures, and distinct medications by generic name are **44, 1, and 14**, respectively. The selected encounter has zero recorded non-laboratory procedures for **30,841 patients (44.06%)**.

In the year before the representative encounter, **51,784/69,990 patients (73.99%)** have zero visits recorded across all three service types. Separately, **9,122** patients have at least one recorded outpatient visit, **5,101** have at least one emergency visit, and **8,195** have at least one inpatient visit. These groups can overlap. All three counts have a cohort-wide median of zero. Among patients with at least one outpatient visit, the median is 1, P99 is 11, and the maximum is 42. See `numeric_profile.csv`, `prior_visit_coverage.csv`, `prior_visit_positive_profile.csv`, `encounter_numeric_distributions.png`, and `prior_visit_distributions.png`.

### Demographics, Medication, and Diagnosis Groups

- **Demographics.** The largest age band is 70 to under 80, with **17,751/69,990 patients (25.36%)**. The recorded `Caucasian` category contains **52,305/69,990 patients (74.73%)**. Age is available in bands. These distributions describe the selected cohort. See `nominal_frequency.csv` and `demographic_distributions.png`.
- **Medication.** Among the 12 retained medication fields, `insulin` has the largest number of patients with a recorded status other than `No` (**35,722**). Only **1 patient** has `chlorpropamide = Down`. The status matrix includes all four statuses and displays zero for an unobserved status. Counts across medications can overlap. See `medication_status.csv` and `medication_distributions.png`.
- **Diagnosis groups.** **68,706/69,990 patients (98.17%)** have mapped diagnosis groups at all three positions. `Circulatory` appears at one or more positions for **40,740/69,990** patients. Each patient is counted once per group in the any-position summary. The fixed ICD-9 groups retain less detail than the source codes. See `diagnosis_summary.csv`, `diagnosis_coverage.csv`, and `diagnosis_group_distributions.png`.

## 6. Descriptive Relationships Between Attributes

| Comparison | Observation | Limit |
| --- | --- | --- |
| Length of stay and medication count | The median is **10** for one-day stays (10,470 patients) and **24** for 14-day stays (642 patients). The table provides Q1 and Q3 for laboratory procedures, non-laboratory procedures, and medications at each stay length. | The measures describe the same representative encounter. The difference does not establish a cause. |
| Age and prior service use | The highest recorded share with at least one emergency visit is in the 20 to under 30 band (**127/1,121**, 11.33%). The highest share with at least one inpatient visit is in the 80 to under 90 band (**1,454/11,110**, 13.09%). | Age-band denominators differ. These counts describe visits recorded in the source and do not establish access to care. |
| Retained medication fields and source flags | Of **53,304** patients with `diabetesMed = Yes`, **34** have no non-`No` status in the 12 retained fields. Of **31,497** patients with `change = Ch`, **14,247** have no `Up` or `Down` status in those fields. | The source flags have broader meanings than these summaries of the 12 retained fields. These combinations alone do not establish a recording error. |

See `stay_service_summary.csv`, `age_prior_visits.csv`, and `medication_flags.csv`.

## 7. Any Readmission Outcome

| Original label | Patients | Share of full cohort | Binary class |
| --- | ---: | ---: | --- |
| `NO` | 41,479 | 59.26% | `not_readmitted` |
| `<30` | 6,285 | 8.98% | `readmitted` |
| `>30` | 22,226 | 31.76% | `readmitted` |

The two recorded readmission categories combine to **28,511/69,990 patients (40.74%)**. `NO` indicates that the source did not record a readmission. The source provides timing categories and no exact number of days. See `outcome_distribution.csv`, `outcome_distributions.png`, and [target_mapping.json](../config/target_mapping.json).

## 8. Questions for Analysis

`eda_findings.csv` summarizes eight findings with their evidence and limits. Questions for a later Analysis include how laboratory-result recording varies across patient groups and how much original code detail the ICD-9 grouping combines. A selected question should specify its population, denominator, and interpretation. These results describe the selected historical cohort and do not establish generalizability to other periods, hospitals, or populations.

## 9. Reproduction and Validation

From the repository root, activate the project environment and install the versions in [requirements.txt](../requirements.txt). Generate the required inputs with [00_data_loading.ipynb](../notebooks/00_data_loading.ipynb) and [01_data_preparation.ipynb](../notebooks/01_data_preparation.ipynb). Run [03_eda.ipynb](../notebooks/03_eda.ipynb) to view the tables, figures, and observations. Generate the local EDA artifacts with:

```bash
python -m src.eda.export
```

`outputs/tables/eda/` contains **24 CSV tables** and `eda_manifest.json`. `outputs/figures/eda/` contains **10 PNG figures**. These generated outputs are ignored by Git and become available locally after the export. The manifest records input and output checksums, cohort scope, statistical conventions, and runtime versions. The verified environment used Python **3.13.5**, pandas **3.1.1**, NumPy **2.5.5**, and Matplotlib **3.11.2**.

At EDA handoff, two independent exports reproduced all 35 files byte for byte in the verified environment. The notebook ran **21 code cells** in a fresh kernel, displayed **10 figures**, and produced in-memory results matching all **24 exported tables**. The **62 Data Preparation checks** and **3 EDA checks** passed. A separate export for this report reproduced all 34 artifacts and the manifest.
