# Readmission Analysis Report

## 1. Purpose and Scope

This analysis examines four questions about the relationships between prior service use, information from a representative inpatient encounter, and recorded readmission. The findings provide reference points for evaluating Any Readmission rules and models in Weka. The intended prediction point is near discharge. [Analysis notebook](../notebooks/04_analysis.ipynb) · [Analysis configuration](../config/analysis.json)

The input is `data/processed/full_prepared.csv`, with **69,990 patients**, one representative encounter per patient, and **32 predictors**. The cohort contains **28,511 positive cases (40.74%)** and **41,479 negative cases (59.26%)**. The positive class combines the source labels `<30` and `>30`. The source label `NO` indicates that no readmission was recorded within the dataset's observation coverage. [Target mapping](../config/target_mapping.json) · [Data Preparation Report](01_data_preparation.md) · [EDA Report](02_eda.md)

The analysis uses the full cohort, including patients assigned to Test. The rates below describe the historical cohort. Record any influence of the analysis on feature, rule, threshold, or model choices, and use development data to evaluate those choices. Using full-cohort results to adjust an approach reduces the independence of the Test evaluation. Report that influence when interpreting the final evaluation.

## 2. Statistical Conventions

Let $N$ be the number of patients in the cohort, $Y$ the number of recorded readmissions, $n$ the number of patients in a group or matching a condition, and $y$ the number of recorded readmissions among them.

| Measure | Calculation | Interpretation |
| --- | --- | --- |
| Coverage | $n/N$ | Share of the cohort in the group or matching the condition |
| Within-group readmission rate | $y/n$ | Share of the group with a recorded readmission |
| Share of positive cases | $y/Y$ | Share of all recorded readmissions contained in the group |
| Rule support | $y/N$ | Share of the cohort matching the condition and having a recorded readmission |
| Rule confidence | $y/n$ | Share of patients matching the condition who have a recorded readmission |
| Lift | $(y/n)/(Y/N)$ | Rule confidence relative to the cohort readmission rate |

Tables flag groups with fewer than 30 patients or fewer than 5 patients in either outcome class. Their counts remain visible. A rate with a zero denominator is undefined and is written as an empty CSV field. Percentage-point differences are calculated before display rounding.

## 3. Q1. How Many Cases Does a History-Based Rule Identify?

Prior service use includes outpatient, emergency, and inpatient visit counts recorded in the year before the representative encounter. The rule raises an alert when at least one visit is recorded across these services. Records missing any of the three counts are reported separately as unknown history. All three counts are complete in the current cohort.

| Recorded history | Patients | Positive cases | Within-group readmission rate | Share of all positive cases |
| --- | ---: | ---: | ---: | ---: |
| All three counts are zero | 51,784 | 18,662 | 36.04% | 65.46% |
| At least one recorded visit | 18,206 | 9,849 | 54.10% | 34.54% |

The rule raises **18,206 alerts**. Of these, **9,849** patients have a recorded readmission and **8,357** have a negative label, giving a precision of **54.10%**. It identifies **9,849** of the cohort's **28,511** positive cases, giving a recall of **34.54%**, and misses **18,662** cases. It correctly identifies **33,122** negative cases. Patients with a recorded prior visit have a higher within-group readmission rate. Patients with all three counts at zero account for nearly two thirds of all positive cases.

For inpatient visits alone, the readmission rates are **38.14% (23,569/61,795)** for zero visits, **55.85% (3,238/5,798)** for one visit, and **71.09% (1,704/2,397)** for two or more visits. This grouping uses one service count. A patient with zero prior inpatient visits can still have outpatient or emergency visits.

The supporting artifacts are `prior_visit_summary.csv`, `prior_visit_combinations.csv` with all eight combinations, `prior_inpatient_levels.csv`, `history_rule_confusion_matrix.csv`, and `prior_visit_summary.png`.

## 4. Q2. How Do Encounter Characteristics Relate to Readmission?

Q2 compares length of stay, diagnosis count, and the ICD-9 group of the first diagnosis within each Q1 history group.

| Length of stay | All three history counts are zero | At least one recorded prior visit |
| --- | ---: | ---: |
| 1–3 days | 33.09% (8,671/26,203) | 51.17% (4,576/8,943) |
| 4–7 days | 38.17% (7,056/18,487) | 56.31% (3,710/6,588) |
| 8–14 days | 41.37% (2,935/7,094) | 58.43% (1,563/2,675) |

Among patients with all three history counts at zero, the rate for stays of 8–14 days is **8.28 percentage points** higher than the rate for stays of 1–3 days. In this same history group, the rate is **40.52% (8,606/21,237)** among patients with nine diagnoses, **38.19% (6,228/16,309)** for the first diagnosis group `Circulatory`, and **29.12% (564/1,937)** for `Neoplasms`. Each diagnosis count from 10 through 16 has fewer than 30 patients in this history group. The first diagnosis group `Other_E` contains one patient.

These comparisons are unadjusted for differences between patient groups. They provide hypotheses for evaluating the predictive value added by encounter variables. The availability of these variables at the intended prediction point requires verification.

The supporting artifacts are `stay_by_history.csv`, `diagnoses_count_by_history.csv`, `primary_diagnosis_by_history.csv`, and `stay_by_history.png`.

## 5. Q3. Which Groups Have Laboratory Results and Treatment Records?

| Recorded laboratory results | Patients | Coverage | Positive cases | Within-group readmission rate |
| --- | ---: | ---: | ---: | ---: |
| Neither result recorded | 54,041 | 77.21% | 22,206 | 41.09% |
| HbA1c only | 12,600 | 18.00% | 4,803 | 38.12% |
| Glucose only | 3,103 | 4.43% | 1,365 | 43.99% |
| Both HbA1c and glucose | 246 | 0.35% | 137 | 55.69% |

A total of **12,846** patients have a recorded HbA1c result and **3,349** have a recorded glucose result. The distribution of result levels uses patients with a recorded result for the corresponding test as its denominator. The readmission rate for a level uses patients in that level. The glucose `>300` group has a readmission rate of **49.86% (355/712)**, compared with **42.15% (717/1,701)** for `Norm`. The literal `None` indicates that no result is recorded. The field does not establish whether a test was performed for an individual patient.

The treatment tables retain the source levels of `insulin`, `change`, and `diabetesMed`. For example, the readmission rate is **45.32% (3,319/7,324)** for insulin `Down` and **38.99% (13,361/34,268)** for `No`. These descriptive differences can reflect patient characteristics. A rule requiring both laboratory results can match at most **246** patients in this cohort. Read its confidence alongside its coverage and the number of positive cases it matches.

The supporting artifacts are `lab_recording_by_outcome.csv`, `lab_results_by_outcome.csv`, `treatment_by_outcome.csv`, and `lab_recording_by_outcome.png`.

## 6. Q4. How Does Combining Conditions Change Confidence?

Both condition pairs include a stay of **8–14 days**. Their respective history conditions are at least one prior inpatient visit and zero recorded visits across all three services. The table shows each individual condition and each intersection. A patient can appear in multiple rows.

| Condition | Patients matched | Positive cases matched | Coverage | Support | Confidence | Lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| At least one prior inpatient visit | 8,195 | 4,942 | 11.71% | 7.06% | 60.31% | 1.48 |
| All three prior service counts are zero | 51,784 | 18,662 | 73.99% | 26.66% | 36.04% | 0.88 |
| Stay of 8–14 days | 9,769 | 4,498 | 13.96% | 6.43% | 46.04% | 1.13 |
| At least one prior inpatient visit and a stay of 8–14 days | 1,578 | 959 | 2.25% | 1.37% | 60.77% | 1.49 |
| All three prior service counts are zero and a stay of 8–14 days | 7,094 | 2,935 | 10.14% | 4.19% | 41.37% | 1.02 |

The first combination has confidence **0.47 percentage points** higher than prior inpatient visits alone and **14.73 points** higher than the stay condition alone. The second combination is **5.33 points** higher than the history condition and **4.67 points** lower than the stay condition. `rule_comparisons.csv` contains six rows because the stay condition appears in both comparisons. Lift uses the full cohort's **40.74%** readmission rate, calculated before rounding.

## 7. Weka Evaluation Reference

Use `analysis_evaluation_map.csv` to trace each finding to its limitation, metrics, and supporting tables. Develop rules or models on Train and perform the comparisons on the same patients in Validation. Recalculate both approaches' metrics on the evaluation set used for the comparison.

| Question | Comparison | Results to examine |
| --- | --- | --- |
| Q1 | Compare the model with the history-based alert | Precision, recall, alert count, false positives, and missed cases. Examine additional cases identified among patients with all three history counts at zero. |
| Q2 | Use the same training approach to compare the three history counts with an approach that adds encounter variables | Recall and false-positive counts among patients with all three history counts at zero. |
| Q3 | Compare model performance by laboratory recording pattern and examine the scope of related rules | Patient and positive-case counts, precision and recall by group, and rule coverage. |
| Q4 | Compare a combined rule mined on Train with each component condition on Validation | Matched patients and positive cases, coverage, support, confidence, and lift using the same formulas. |

Evaluate the predictive contribution of a variable or combination on evaluation data. The current comparisons describe a cohort with one representative encounter per patient. The smallest eligible ID makes selection reproducible. The source lacks timestamps to establish the first admission, exact follow-up duration, or readmissions outside the recorded coverage. [Data preparation limitations](01_data_preparation.md#11-methodological-limitations)

## 8. Reproduction and Validation

Generate `full_prepared.csv` and the preparation manifest through the data preparation workflow. Open the [Analysis notebook](../notebooks/04_analysis.ipynb) with the project kernel to view the results. From the repository root with the Python environment active, run the following command.

```bash
python -m src.analysis.export
```

The command creates **12 CSV files and one manifest** in `outputs/tables/analysis/` and **3 PNG files** in `outputs/figures/analysis/`. These outputs are generated locally and ignored by Git. Use `--output-dir` for a separate destination. Use `--overwrite` to replace files with different contents after reviewing the change. `analysis_manifest.json` records source paths, SHA-256 checksums, configuration, runtime versions, and statistical denominators.

The verified environment used Python **3.13.5**, pandas **3.1.1**, NumPy **2.5.5**, Matplotlib **3.11.2**, and FreeType **2.14.3**. Two independent CLI runs reproduced all **16 files byte for byte**. The notebook ran **12 code cells** in a fresh kernel, displayed **3 figures**, and produced **12 tables matching the exported CSVs**. Independent counts verified **18 rows** across Q1–Q4. All **81 repository checks** passed at the Analysis handoff. A separate export for this report also matched all 16 files at the default location. [Python requirements](../requirements.txt) · [Analysis checks](../tests/analysis/)
