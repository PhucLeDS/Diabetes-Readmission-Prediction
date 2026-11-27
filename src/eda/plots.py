"""Plot descriptive distributions for the full prepared cohort."""

from textwrap import shorten

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from src.eda.data import ADMISSION_COLUMNS
from src.eda.descriptive import PRIOR_VISIT_COLUMNS
from src.eda.information_coverage import LAB_RECORDING_PATTERNS, InformationCoverage
from src.eda.nominal import DEMOGRAPHIC_COLUMNS, MEDICATION_FLAG_COLUMNS, NominalProfile
from src.eda.relationships import STAY_COLUMN, STAY_SERVICE_COLUMNS, RelationshipProfile
from src.preparation.dataset_preparation import DIAGNOSIS_GROUP_COLUMNS


ENCOUNTER_MEASURES = {
    "time_in_hospital": ("Time in hospital", "Days"),
    "num_lab_procedures": ("Laboratory procedures", "Laboratory tests"),
    "num_procedures": ("Other procedures", "Procedures"),
    "num_medications": ("Medications", "Distinct medications"),
    "number_diagnoses": ("Diagnoses", "Diagnoses entered"),
}
PRIOR_VISIT_LABELS = {
    "number_outpatient": "Outpatient",
    "number_emergency": "Emergency",
    "number_inpatient": "Inpatient",
}
BAR_COLOR = "#286b8f"


def plot_numeric_distributions(frequency: pd.DataFrame, *, cohort_size: int) -> dict[str, Figure]:
    """Build reusable figures from the aggregate frequency table."""
    # Keep inline and standalone rendering on the same figure settings.
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        current, current_axes = plt.subplots(2, 3, figsize=(12, 6.6), layout="constrained")
        current.suptitle(f"Representative Inpatient Encounter | Full prepared cohort (N = {cohort_size:,})")
        for ax, (column, (title, unit)) in zip(current_axes.flat, ENCOUNTER_MEASURES.items()):
            counts = frequency.loc[frequency["column"].eq(column)]
            ax.bar(counts["value"], counts["patients"], color=BAR_COLOR, width=0.85)
            ax.set(title=title, xlabel=unit, ylabel="Patients")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        current_axes.flat[-1].remove()

        prior, prior_axes = plt.subplots(3, 2, figsize=(11, 9), layout="constrained")
        prior.suptitle(f"Service Use in the Prior Year | Full prepared cohort (N = {cohort_size:,})")
        for (coverage_ax, distribution_ax), column in zip(prior_axes, PRIOR_VISIT_COLUMNS, strict=True):
            counts = frequency.loc[frequency["column"].eq(column)]
            valid = int(counts["patients"].sum())
            zero = int(counts.loc[counts["value"].eq(0), "patients"].sum())
            positive = counts.loc[counts["value"].gt(0)]
            bar_counts = [zero, valid - zero]
            bars = coverage_ax.barh(["0 visits", "1+ visits"], bar_counts, color=["#8295a0", BAR_COLOR])
            coverage_ax.bar_label(bars, labels=[f"{n:,}" for n in bar_counts], padding=4)
            coverage_ax.invert_yaxis()
            coverage_ax.set(
                title=f"{PRIOR_VISIT_LABELS[column]} | {valid:,} valid",
                xlabel="Patients",
                xlim=(0, max(valid, 1) * 1.2),
            )

            distribution_ax.bar(positive["value"], positive["patients"], width=0.8, color=BAR_COLOR)
            distribution_ax.set(
                title=f"{PRIOR_VISIT_LABELS[column]} | {valid - zero:,} with recorded visits",
                xlabel="Number of visits",
                ylabel="Patients (log scale)" if not positive.empty else "Patients",
            )
            if not positive.empty:
                distribution_ax.set_yscale("log")
            distribution_ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))

    return {"encounter_numeric_distributions": current, "prior_visit_distributions": prior}


def _plot_category_counts(ax, rows: pd.DataFrame, *, title: str, labels=None, colors=BAR_COLOR) -> None:
    labels = rows["value"].replace("", "(Empty field)") if labels is None else labels
    counts = rows["patients"]
    bars = ax.barh(labels, counts, color=colors)
    ax.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=4, fontsize=8)
    ax.invert_yaxis()
    ax.set(title=title, xlabel="Patients", xlim=(0, max(int(counts.max()), 1) * 1.23))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))


def plot_nominal_distributions(profile: NominalProfile) -> dict[str, Figure]:
    """Build one cohort figure per group of nominal predictors."""
    frequency = profile.nominal_frequency
    cohort_size = int(frequency["cohort_size"].iloc[0])
    scope = f"Full prepared cohort (N = {cohort_size:,})"
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        demographics, axes = plt.subplots(1, 3, figsize=(12, 4.3), layout="constrained")
        demographics.suptitle(f"Demographics | {scope}")
        for ax, column in zip(axes, DEMOGRAPHIC_COLUMNS, strict=True):
            _plot_category_counts(ax, frequency.loc[frequency["column"].eq(column)], title=column.title())

        admissions, axes = plt.subplots(1, 2, figsize=(15, 7.5), layout="constrained")
        admissions.suptitle(f"Admission Codes | {scope}")
        for ax, column, title in zip(
            axes, ADMISSION_COLUMNS, ("Admission type", "Admission source"), strict=True,
        ):
            rows = profile.admission_descriptions.loc[profile.admission_descriptions["column"].eq(column)]
            labels = rows["code"].replace("", "(Empty field)") + " | " + rows["description"].where(
                rows["description"].ne(""), rows["lookup_status"],
            )
            labels = [shorten(label, width=42, placeholder="...") for label in labels]
            colors = rows["lookup_status"].map(lambda status: BAR_COLOR if status == "Mapped" else "#ad703b")
            _plot_category_counts(ax, rows, title=title, labels=labels, colors=colors)

        lab, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
        lab.suptitle(f"Laboratory Results | {scope}")
        lab_titles = {"A1Cresult": "HbA1c", "max_glu_serum": "Serum glucose"}
        recorded_by_test = profile.laboratory_results.drop_duplicates("column").set_index("column")
        share = recorded_by_test.loc[list(lab_titles), "recorded_percent_of_cohort"]
        bars = axes[0].barh(list(lab_titles.values()), share, color=BAR_COLOR)
        axes[0].bar_label(bars, labels=[f"{percent:.2f}%" for percent in share], padding=4, fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set(title="Recorded result by test", xlabel="% of full cohort", xlim=(0, 100))
        for ax, (column, title) in zip(axes[1:], lab_titles.items(), strict=True):
            rows = profile.laboratory_results.loc[
                profile.laboratory_results["column"].eq(column) & profile.laboratory_results["result_recorded"]
            ]
            n = int(rows["patients"].sum())
            if rows.empty:
                ax.set(title=f"{title} result levels (n = 0)")
            else:
                _plot_category_counts(ax, rows, title=f"{title} result levels (n = {n:,})")

        medications, axes = plt.subplots(1, 2, figsize=(12, 6.4), width_ratios=(1.45, 1), layout="constrained")
        medications.suptitle(f"Diabetes Medication Records | {scope}")
        drugs = profile.medication_status.sort_values("any_non_no_patients", ascending=False, kind="stable")
        counts = drugs["any_non_no_patients"]
        axes[0].scatter(counts, drugs["column"], color=BAR_COLOR, s=26)
        for count, drug in zip(counts, drugs["column"], strict=True):
            axes[0].annotate(f"{count:,}", (count, drug), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
        axes[0].invert_yaxis()
        # Symmetric log keeps a zero count visible if a drug has no recorded non-No status.
        axes[0].set_xscale("symlog", linthresh=1)
        axes[0].set(
            title="Patients with a recorded non-No status",
            xlabel="Patients (logarithmic above 1)",
            xlim=(0, max(int(counts.max()), 10) * 2.2),
        )
        axes[0].grid(axis="x", alpha=0.22)
        flags = frequency.loc[frequency["column"].isin(MEDICATION_FLAG_COLUMNS)]
        flag_labels = flags["column"].map({"change": "Change", "diabetesMed": "Diabetes medication"})
        flag_labels = flag_labels + " | " + flags["value"].replace("", "(Empty field)")
        _plot_category_counts(axes[1], flags, title="Summary flags", labels=flag_labels)

        diagnoses, axes = plt.subplots(1, 2, figsize=(13.5, 7), sharey=True, layout="constrained")
        diagnoses.suptitle(f"Diagnosis Groups | {scope}")
        groups = profile.diagnosis_summary
        positions = np.arange(len(groups))
        for level, (column, title, color) in enumerate(zip(
            DIAGNOSIS_GROUP_COLUMNS, ("Primary", "Secondary", "Additional secondary"),
            (BAR_COLOR, "#459787", "#8295a0"), strict=True,
        )):
            axes[0].barh(
                positions + (level - 1) * 0.24, groups[f"{column}_patients"],
                height=0.23, label=title, color=color,
            )
        axes[0].set_yticks(positions, labels=groups["group"].replace("", "(Empty field)"))
        axes[0].invert_yaxis()
        axes[0].set(title="By position", xlabel="Patients")
        axes[0].legend(fontsize=8, loc="lower right")
        bars = axes[1].barh(positions, groups["patients_at_any_position"], color=BAR_COLOR, height=0.70)
        axes[1].bar_label(
            bars, labels=[f"{n:,}" for n in groups["patients_at_any_position"]], padding=4, fontsize=8,
        )
        axes[1].set(
            title="At any position (one count per patient and group)", xlabel="Patients",
            xlim=(0, max(int(groups["patients_at_any_position"].max()), 1) * 1.2),
        )

    return {
        "demographic_distributions": demographics,
        "admission_distributions": admissions,
        "laboratory_result_distributions": lab,
        "medication_distributions": medications,
        "diagnosis_group_distributions": diagnoses,
    }



def _plot_recording_patterns(
    table: pd.DataFrame, *, group: str, labels: dict[str, str], title: str, cohort_size: int,
) -> Figure:
    observed = table.loc[table["group_patients"].gt(0)]
    denominators = observed.drop_duplicates(group).set_index(group)["group_patients"]
    proportions = observed.pivot(index=group, columns="lab_recording_pattern", values="percent_of_group")
    proportions = proportions.reindex(index=denominators.index, columns=LAB_RECORDING_PATTERNS)
    styles = {
        "neither": ("Neither result", "#a5acb4"),
        "a1c_only": ("HbA1c only", BAR_COLOR),
        "glucose_only": ("Glucose only", "#c88424"),
        "both": ("Both results", "#308475"),
        "unknown": ("Unknown", "#815e8a"),
    }
    positions = np.arange(len(denominators))
    figure, ax = plt.subplots(
        figsize=(11, max(3.5, 2 + 0.43 * len(denominators))), layout="constrained",
    )
    figure.suptitle(f"{title} | Full prepared cohort (N = {cohort_size:,})")
    left = np.zeros(len(positions))
    for pattern, (label, color) in styles.items():
        widths = proportions[pattern].to_numpy(dtype=float)
        if widths.any():
            ax.barh(positions, widths, left=left, label=label, color=color)
            left += widths
    ax.set_yticks(positions, labels=[
        f"{labels.get(value, value)} (n = {count:,})" for value, count in denominators.items()
    ])
    ax.invert_yaxis()
    ax.set(xlabel="Patients in group (%)", xlim=(0, 100))
    figure.legend(loc="outside lower center", ncols=5, frameon=False)
    return figure


def plot_information_coverage(
    profile: InformationCoverage, admission_descriptions: pd.DataFrame,
) -> dict[str, Figure]:
    """Plot subgroup shares from the same tables used for the EDA export."""
    history = profile.lab_recording_by_prior_history
    cohort_size = int(history["patients"].sum())
    types = profile.lab_recording_by_characteristic.loc[
        profile.lab_recording_by_characteristic["characteristic"].eq("admission_type_id")
    ]
    lookup = admission_descriptions.loc[admission_descriptions["column"].eq("admission_type_id")]
    type_labels = {
        row.code: shorten(f"{row.code} | {row.description or row.lookup_status}", width=42, placeholder="...")
        for row in lookup.itertuples(index=False)
    }
    type_labels[""] = "(Empty field)"
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        by_history = _plot_recording_patterns(
            history, group="prior_history", labels={
                "no_recorded_prior_visits": "Zero recorded visits",
                "any_recorded_prior_visit": "At least one recorded visit",
                "unknown": "Unknown history",
            }, title="Laboratory Recording by Prior Service Use", cohort_size=cohort_size,
        )
        by_admission = _plot_recording_patterns(
            types, group="value", labels=type_labels,
            title="Laboratory Recording by Admission Type", cohort_size=cohort_size,
        )
    return {
        "lab_recording_by_prior_history": by_history,
        "lab_recording_by_admission_type": by_admission,
    }


def plot_relationships(profile: RelationshipProfile) -> dict[str, Figure]:
    """Plot stay summaries, prior visit proportions, and outcome totals."""
    cohort_size = int(profile.outcome_distribution["cohort_size"].iloc[0])
    scope = f"Full prepared cohort (N = {cohort_size:,})"
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        stays, axes = plt.subplots(1, 3, figsize=(12, 4.3), layout="constrained")
        stays.suptitle(f"Services by Length of Stay | {scope}")
        for ax, measure in zip(axes, STAY_SERVICE_COLUMNS, strict=True):
            rows = profile.stay_service_summary.loc[profile.stay_service_summary["measure"].eq(measure)]
            observed = rows.loc[rows[STAY_COLUMN].notna()]
            days = observed[STAY_COLUMN].to_numpy(dtype=float)
            median = observed["median"].to_numpy(dtype=float, na_value=np.nan)
            q1 = observed["q1"].to_numpy(dtype=float, na_value=np.nan)
            q3 = observed["q3"].to_numpy(dtype=float, na_value=np.nan)
            ax.fill_between(days, q1, q3, color=BAR_COLOR, alpha=0.17, label="Q1 to Q3")
            ax.plot(days, median, marker="o", markersize=3, linewidth=1.5, color=BAR_COLOR, label="Median")
            title, unit = ENCOUNTER_MEASURES[measure]
            ax.set(title=title, xlabel="Days in hospital", ylabel=unit)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
            ax.legend(loc="best", fontsize=8)

        visits, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
        visits.suptitle(f"Recorded Prior Visits by Age Band | {scope}")
        age_labels = profile.age_prior_visits["age"].drop_duplicates().replace("", "(Empty field)")
        positions = np.arange(len(age_labels))
        for service, color in zip(PRIOR_VISIT_COLUMNS, (BAR_COLOR, "#9a6336", "#35816f"), strict=True):
            rates = profile.age_prior_visits.loc[
                profile.age_prior_visits["service"].eq(service), "percent_of_age_group"
            ]
            ax.plot(positions, rates, marker="o", linewidth=1.6, label=PRIOR_VISIT_LABELS[service], color=color)
        ax.set_xticks(positions, labels=age_labels, rotation=30, ha="right")
        ax.set(xlabel="Age band (years)", ylabel="Patients with 1+ visits (% of age band)", ylim=(0, None))
        ax.legend(loc="best", fontsize=8)
        ax.grid(axis="y", alpha=0.22)

        outcome, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
        outcome.suptitle(f"Recorded Readmission Labels | {scope}")
        for ax, representation in zip(axes, ("Original", "Any Readmission"), strict=True):
            rows = profile.outcome_distribution.loc[
                profile.outcome_distribution["representation"].eq(representation)
            ]
            _plot_category_counts(ax, rows, labels=rows["label"], title=representation)
            ax.set_xlim(0, cohort_size)

    return {
        "stay_service_distributions": stays,
        "age_prior_visit_distributions": visits,
        "outcome_distributions": outcome,
    }
