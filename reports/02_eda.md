# Exploratory Data Analysis Report

## 1. Purpose and Scope

This report describes the prepared dataset for the Any Readmission prediction task. It examines cohort selection, data completeness, numeric and nominal attributes, recorded treatment, diagnosis coverage, and relationships between attributes.

The prepared cohort contains **69,990 patients**, with one representative inpatient encounter per patient. The primary input, `data/processed/full_prepared.csv`, has 35 columns comprising two identifiers, **32 predictors** split into 8 numeric and 24 nominal attributes, and the original `readmitted` label. The EDA also reads the cleaned encounters, selected source cohort, lookup, configuration and preparation audit to explain selection and compare retained features with their source columns. `column_dictionary.csv` describes the role, source, domain, and period associated with each prepared field. [EDA notebook](../notebooks/03_eda.ipynb) · [Data Preparation Report](01_data_preparation.md)

The EDA uses the **full prepared cohort, including Test**. It describes the available historical records and summarizes readmission labels at cohort level. Record any influence of these full-cohort findings on feature, preprocessing, rule, or model choices when interpreting the final Test evaluation. The dataset does not establish generalizability to other periods, hospitals, or populations.

## 2. Cohort Selection

| Stage | Encounters | Unique patients | Encounters removed at this stage |
| --- | ---: | ---: | ---: |
| Cleaned, with the same rows as the raw source | 101,766 | 71,518 | 0 |
| Eligible after discharge-code filtering | 99,343 | 69,990 | 2,423 |
| One representative encounter per patient | 69,990 | 69,990 | 29,353 |

The pipeline excludes encounters with the configured death or hospice discharge codes, then selects the eligible encounter with the smallest numeric `encounter_id` for each patient. A patient may retain an eligible encounter even if another encounter was excluded. The source has no encounter timestamps to establish whether the selected encounter was the patient's first admission. The prepared cohort retains **3,252 encounters** whose discharge descriptions are unclear.

Among the 71,518 patients in the cleaned data, **871** have both eligible and excluded encounters, and **1,528** have only excluded encounters. Within the eligible population, **16,341/69,990 patients (23.35%)** have more than one eligible encounter. They contribute 16,341 representative and 29,353 other eligible encounters.

The age band 80 to under 90 represents **763/2,423 excluded encounters (31.49%)** and **16,434/99,343 eligible encounters (16.54%)**. These percentages use encounters as the denominator, so patients with multiple encounters can contribute more than once. Among patients with multiple eligible encounters, one-day stays account for **2,048/16,341 representative encounters (12.53%)** and **3,354/29,353 other eligible encounters (11.43%)**. The second group gives more weight to patients with more encounters.

See `cohort_flow.csv`, `discharge_patient_overlap.csv`, `discharge_eligibility_profile.csv`, `eligible_encounters_per_patient.csv`, and `representative_encounter_profile.csv`.

## 3. Statistical Conventions

- **Numeric attributes.** The profile reports valid and missing counts, zeros, minimum, Q1, median, Q3, P95, P99, and maximum. Quantiles use linear interpolation over valid observations. The zero percentage uses the valid count for that attribute. The frequency table retains every observed value.
- **Nominal attributes.** Frequencies and percentages use the full cohort. A rare level is an observed nonempty category present in less than 1% of the cohort. Source codes, rare values, and empty fields remain visible.
- **Group denominators.** Selection comparisons identify whether their unit is an encounter or a patient. Laboratory coverage uses the relevant cohort or subgroup, while the distribution of a recorded result uses patients with a result for that test. Admission source percentages use the corresponding admission type. Medication pair percentages use the insulin group or the full cohort, as labeled.
- **Small and unknown groups.** The detailed selection, admission, laboratory coverage, diagnosis, and medication tables flag cells with fewer than 30 observations where specified in the manifest. Unknown laboratory recording status covers an empty or unexpected value. A percentage with a zero denominator is left empty.
- **Repeated membership.** A patient can have more than one medication or diagnosis group. Pair counts include each patient once per pair, and different pair counts can overlap.

The manifest records the denominators and rules for each output. [EDA exporter](../src/eda/export.py)

## 4. Completeness and Source Categories

The prepared CSV contains **1,922 empty fields across 1,920 patients (2.74%)**. There are 1,919 empty `race` fields and 3 empty `gender` fields. Two patients are missing both. The eight numeric attributes have no empty fields in this cohort.

The diagnosis groups use `Missing` for an empty source code and `Unmapped` for a code outside the grouping rules. The counts of `Missing` at the three diagnosis positions are **10, 293, and 1,224**, respectively. No `Unmapped` values appear in the prepared cohort.

`admission_type_id` and `admission_source_id` use separate lookup tables. Every observed code has a lookup entry. Descriptions are unclear for **7,893/69,990 admission types (11.28%)** and **5,069/69,990 admission sources (7.24%)**. These are column-specific counts, and a patient can appear in both groups. See `missing_summary.csv`, `missing_patterns.csv`, `special_values.csv`, and `admission_descriptions.csv`.

### Laboratory Results

| Test | Patients with a recorded result | Share of 69,990 patients | Most frequent recorded level |
| --- | ---: | ---: | --- |
| HbA1c | 12,846 | 18.35% | `>8` in 6,239/12,846 patients (48.57%) |
| Serum glucose | 3,349 | 4.78% | `Norm` in 1,701/3,349 patients (50.79%) |

A total of **246** patients have both results recorded, **12,600** have only HbA1c, **3,103** have only serum glucose, and **54,041/69,990 (77.21%)** have neither result recorded. Neither laboratory column contains an empty CSV field in this cohort. The literal `None` indicates no result was recorded. The source does not establish whether a test was performed. The EDA rules identify empty or unexpected laboratory values as an unknown recording status.

Among patients with zero recorded prior visits, **39,726/51,784 (76.71%)** have neither result recorded. The corresponding count is **14,315/18,206 (78.63%)** among patients with at least one recorded prior visit. Prior visits cover outpatient, emergency, and inpatient services in the preceding year.

Admission type code `6`, whose lookup description is `NULL`, accounts for **215/246 patients (87.40%)** with both laboratory results recorded. Within the code `6` group, **215/4,516 patients (4.76%)** have both results recorded. The first percentage describes concentration among patients with both results. The second describes coverage within the admission group. See `lab_recording_patterns.csv`, `laboratory_results.csv`, `lab_recording_by_prior_history.csv`, `lab_recording_by_characteristic.csv`, and the laboratory recording figures.

## 5. Cohort Characteristics

### Current Encounter and Prior Service Use

The median length of stay is **3 days**, with Q1 at 2 days and Q3 at 6 days. The median numbers of laboratory procedures, non-laboratory procedures, and distinct medications by generic name are **44, 1, and 14**, respectively. The selected encounter has zero recorded non-laboratory procedures for **30,841 patients (44.06%)**.

In the year before the representative encounter, **51,784/69,990 patients (73.99%)** have zero visits recorded across all three service types. Separately, **9,122** patients have at least one recorded outpatient visit, **5,101** have at least one emergency visit, and **8,195** have at least one inpatient visit. These groups can overlap. All three counts have a cohort-wide median of zero. Among patients with at least one outpatient visit, the median is 1, P99 is 11, and the maximum is 42.

For `number_diagnoses`, **30,719/69,990 patients (43.89%)** have exactly nine diagnoses recorded and **73/69,990 (0.10%)** have more than nine. The source provides no explanation for the concentration at nine. See `numeric_profile.csv`, `numeric_frequency.csv`, `prior_visit_coverage.csv`, `prior_visit_positive_profile.csv`, and the numeric distribution figures.

### Demographics and Admission Context

The largest age band is 70 to under 80, with **17,751/69,990 patients (25.36%)**. The recorded `Caucasian` category contains **52,305/69,990 patients (74.73%)**. Age is available in bands, and these distributions describe the selected cohort.

The Emergency Room is the recorded source for **31,456/35,480 Emergency admissions (88.66%)**. The median number of non-laboratory procedures is **0** for Emergency admissions and **2** for Elective admissions. Type and source describe different aspects of the encounter. Codes with unclear descriptions and small groups need to be interpreted with their counts. See `nominal_frequency.csv`, `admission_type_source.csv`, `age_by_admission.csv`, `numeric_by_admission.csv`, and `admission_type_source.png`.

### Medication Records

Among the 12 retained medication fields, `insulin` has the largest number of patients with a recorded status other than `No` (**35,722**). Only **1 patient** has `chlorpropamide = Down`. In total, **22,129/69,990 patients (31.62%)** have a status of `Steady`, `Up`, or `Down` for at least two retained medications. Metformin is recorded alongside insulin for **7,298/35,722 patients (20.43%)** with recorded insulin. Medication counts can overlap and do not establish treatment timing.

Of **53,304** patients with `diabetesMed = Yes`, **34** have no recorded non-`No` status in the retained fields. All **34/34** have a matching status in the excluded source medication fields. Of **31,497** patients with `change = Ch`, **14,247** have no `Up` or `Down` status in the retained fields, and **1/14,247** has a matching status in the excluded fields. These comparisons describe the recorded fields and do not establish why the flags were set.

See `medication_status.csv`, `medication_count_distribution.csv`, `insulin_medication_pairs.csv`, `medication_flags.csv`, `medication_flag_reconciliation.csv`, and `medication_distributions.png`.

### Diagnosis Groups and Source Codes

**68,706/69,990 patients (98.17%)** have mapped diagnosis groups at all three positions. Of these, **7,360** have one distinct group, **31,070** have two, and **30,276** have three. Thus **38,430/68,706 (55.93%)** have a repeated group.

The `Circulatory` group appears at one or more positions for **40,740/69,990 patients (58.21%)** and combines **57 distinct source codes**. Each patient is counted once per group in the any-position summary. A broad group can combine clinically distinct codes.

The `Diabetes` group appears in the primary position for **5,748** patients and at any of the three positions for **27,578**. The additional **21,830** patients represent **31.19 percentage points** of the full cohort. This measures the coverage of the recorded diagnosis group within a cohort already selected from patients with diabetes. `Circulatory` and `Diabetes` occur together in **13,607/69,990 patients (19.44%)**. A patient can contribute to multiple group pairs.

See `diagnosis_summary.csv`, `diagnosis_coverage.csv`, `diagnosis_position_depth.csv`, `diagnosis_group_gain.csv`, `diagnosis_group_pairs.csv`, `diagnosis_source_code_summary.csv`, `diagnosis_source_code_detail.csv`, and the diagnosis group figures.

## 6. Descriptive Relationships Between Attributes

| Comparison | Observation | Interpretation limit |
| --- | --- | --- |
| Length of stay and medication count | The median is **10** for one-day stays (10,470 patients) and **24** for 14-day stays (642 patients). Q1 and Q3 are available for laboratory procedures, non-laboratory procedures, and medications at each stay length. | The measures describe the same representative encounter. The contrast does not establish a cause. |
| Age and prior service use | The highest recorded share with at least one emergency visit is in the 20 to under 30 band (**127/1,121**, 11.33%). The highest share with at least one inpatient visit is in the 80 to under 90 band (**1,454/11,110**, 13.09%). | Age-band denominators differ. Recorded visits do not establish access to care. |
| Admission type and source | **31,456/35,480 Emergency admissions (88.66%)** record the Emergency Room as their source. | This share is conditional on Emergency admission type. The table also retains less frequent and unclear codes. |
| Laboratory recording and prior service use | Neither result is recorded for **39,726/51,784 (76.71%)** with zero prior visits and **14,315/18,206 (78.63%)** with at least one prior visit. | These proportions describe recorded results within each history group. They do not establish why results are absent. |

See `stay_service_summary.csv`, `age_prior_visits.csv`, `admission_type_source.csv`, and `lab_recording_by_prior_history.csv`.

## 7. Any Readmission Outcome

| Original label | Patients | Share of full cohort | Binary class |
| --- | ---: | ---: | --- |
| `NO` | 41,479 | 59.26% | `not_readmitted` |
| `<30` | 6,285 | 8.98% | `readmitted` |
| `>30` | 22,226 | 31.76% | `readmitted` |

The two recorded readmission categories combine to **28,511/69,990 patients (40.74%)**. `NO` indicates that the source did not record a readmission within its observation coverage. The source provides timing categories and no exact number of days. See `outcome_distribution.csv`, `outcome_distributions.png`, and [target mapping](../config/target_mapping.json).

## 8. Directions for Readmission Analysis

`eda_findings.csv` contains **18 findings** with evidence, interpretation, limitations, and source tables. `eda_analysis_directions.csv` connects those findings to four directions for a single integrated readmission results section.

| Direction | Analysis scope and contribution |
| --- | --- |
| Cohort composition and outcome coverage | Establish the patient-level baseline. Explain the effects of discharge exclusions and representative selection using the appropriate encounter and patient denominators. Describe unclear discharge information. |
| Prior service use and admission context | Compare readmission across prior service use, age, and admission context. Report each group's rate together with its share of all recorded readmissions. |
| Laboratory recording and treatment documentation | Examine result coverage and recorded treatment alongside prior use and admission type. Report subgroup sizes, unknown values, sparse statuses, and the limits of absent treatment timing. |
| Diagnosis burden and coding coverage | Compare diagnosis counts, positions, and common co-occurring groups. Track missing positions and overlapping patient groups when interpreting readmission patterns. |

Synthesize the results across these directions and report both group-specific rates and the number of recorded cases represented. The existing [Analysis notebook](../notebooks/04_analysis.ipynb) and [Analysis Report](03_analysis.md) contain earlier readmission analyses. The directions above identify further comparisons to consider in a later Analysis revision.

## 9. Reproduction and Validation

From the repository root, activate the project environment and install the pinned packages in [requirements.txt](../requirements.txt). Place both verified raw source files in `data/raw/`, then run [00_data_loading.ipynb](../notebooks/00_data_loading.ipynb) and [01_data_preparation.ipynb](../notebooks/01_data_preparation.ipynb) to create the cleaned and cohort data, prepared CSV, feature audit, and preparation manifest. The EDA also reads the admission lookup and the preparation and target-mapping configurations.

Run [03_eda.ipynb](../notebooks/03_eda.ipynb) to view the results. Generate the local tables, figures, and manifest from the repository root with:

```bash
python -m src.eda.export
```

`outputs/tables/eda/` contains **42 CSV tables** and `eda_manifest.json`. `outputs/figures/eda/` contains **14 PNG figures**. Git ignores these generated outputs, which can be recreated with the export command. The manifest records input and artifact checksums, cohort scope, statistical conventions, and runtime versions. Use `--output-dir` for a separate export, or `--overwrite` after reviewing an intended replacement of existing artifacts.

An independent EDA export reproduced all **56 tables and figures** byte for byte against the local artifacts, and its manifest matched. The EDA notebook executed **30 code cells** in a fresh kernel, displayed **14 figures**, and produced both handoff tables with the same values as the export. All **17 EDA tests** passed. The verified environment used Python **3.13.5**, pandas **3.1.1**, NumPy **2.5.5**, Matplotlib **3.11.2**, and FreeType **2.14.3**. [EDA tests](../tests/eda/)
