"""Plot aggregate results from the readmission analysis."""

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import pandas as pd

from src.analysis.current_encounters import MISSING_STAY, OUTSIDE_STAY_RANGE


HISTORY_LABELS = {
    "no_recorded_prior_visits": "No recorded prior visits",
    "any_recorded_prior_visit": "Any recorded prior visit",
    "unknown": "Unknown history",
}


def plot_prior_visit_summary(summary: pd.DataFrame) -> Figure:
    """Show each history group's readmission rate and count on separate scales."""
    labels = summary["history_group"].map(HISTORY_LABELS)
    counts = summary["readmissions"]
    rates = summary["readmission_rate_percent"]
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        figure, (rate_ax, count_ax) = plt.subplots(1, 2, figsize=(10.4, 3.5), layout="constrained")
        rate_bars = rate_ax.barh(labels, rates.fillna(0), color="#2F6786")
        rate_ax.bar_label(rate_bars, labels=[f"{value:.1f}%" if pd.notna(value) else "Undefined" for value in rates],
                          padding=4)
        rate_ax.set(xlabel="Share of patients in group", title="Readmission rate within each group", xlim=(0, 115))
        rate_ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))

        count_bars = count_ax.barh(labels, counts, color="#2F6786")
        count_ax.bar_label(count_bars, labels=[f"{value:,}" for value in counts], padding=4)
        count_ax.set(xlabel="Recorded readmissions", title="Readmission count by group",
                     xlim=(0, max(1, counts.max() * 1.24)))
        count_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        for ax in (rate_ax, count_ax):
            ax.invert_yaxis()
        cohort_size = summary["cohort_size"].iloc[0]
        cohort_readmissions = summary["cohort_readmissions"].iloc[0]
        figure.suptitle(f"Prior service use and Any Readmission\nFull cohort of {cohort_size:,} patients with "
                        f"{cohort_readmissions:,} recorded readmissions")
    return figure


def plot_stay_by_history(summary: pd.DataFrame) -> Figure:
    """Compare stay bands by prior history on a shared readmission rate axis."""
    bands = summary.loc[~summary["stay_group"].isin([MISSING_STAY, OUTSIDE_STAY_RANGE])]
    stay_order = bands["stay_group"].drop_duplicates().tolist()
    history_order = bands["history_group"].drop_duplicates().tolist()
    colors = {
        "no_recorded_prior_visits": "#2F6786",
        "any_recorded_prior_visit": "#A8501F",
        "unknown": "#6B6B6B",
    }
    width = 0.78 / len(history_order)

    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        figure, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
        for index, history in enumerate(history_order):
            rows = bands.loc[bands["history_group"].eq(history)].set_index("stay_group").loc[stay_order]
            rates, patients = rows["readmission_rate_percent"], rows["patients"]
            offset = (index - (len(history_order) - 1) / 2) * width
            bars = ax.bar(
                [position + offset for position in range(len(stay_order))], rates.fillna(0),
                width=width, label=HISTORY_LABELS[history], color=colors[history],
            )
            labels = [
                f"{rate:.1f}%\n(n={count:,})" if pd.notna(rate) else "Undefined\n(n=0)"
                for rate, count in zip(rates, patients, strict=True)
            ]
            ax.bar_label(bars, labels=labels, padding=4, fontsize=9)
        ax.set(
            xlabel="Days in hospital", ylabel="Recorded readmissions within group",
            ylim=(0, 114), yticks=range(0, 101, 20),
        )
        ax.set_xticks(range(len(stay_order)), stay_order)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(history_order), frameon=False)
        cohort_size = summary["cohort_size"].iloc[0]
        figure.suptitle(f"Length of stay and Any Readmission by prior service use\nFull cohort of {cohort_size:,} patients")
    return figure


def plot_lab_recording_by_outcome(summary: pd.DataFrame) -> Figure:
    """Compare laboratory recording coverage and readmission rates by pattern."""
    labels = summary["recording_pattern"].map({
        "neither": "Neither result recorded",
        "a1c_only": "HbA1c only",
        "glucose_only": "Glucose only",
        "both": "Both results recorded",
        "unknown": "Unknown recording status",
    })
    rates = summary["readmission_rate_percent"]
    coverage = summary["coverage_percent"]
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        figure, (coverage_ax, rate_ax) = plt.subplots(1, 2, figsize=(10.8, 4.0), sharey=True, layout="constrained")
        coverage_bars = coverage_ax.barh(labels, coverage.fillna(0), color="#2F6786")
        coverage_ax.bar_label(
            coverage_bars,
            labels=[f"{rate:.1f}% (n={count:,})" for rate, count in zip(coverage, summary["patients"], strict=True)],
            padding=4, fontsize=9,
        )
        coverage_ax.set(title="Coverage in the full cohort", xlabel="Share of all patients")

        rate_bars = rate_ax.barh(labels, rates.fillna(0), color="#A8501F")
        rate_ax.bar_label(
            rate_bars,
            labels=[f"{rate:.1f}%" if pd.notna(rate) else "Undefined" for rate in rates],
            padding=4, fontsize=9,
        )
        rate_ax.set(title="Readmission rate within each group", xlabel="Share of patients in group")
        for ax in (coverage_ax, rate_ax):
            ax.set(xlim=(0, 115), xticks=range(0, 101, 20))
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
        coverage_ax.invert_yaxis()
        cohort_size = summary["cohort_size"].iloc[0]
        figure.suptitle(f"Laboratory result recording and Any Readmission\nFull cohort of {cohort_size:,} patients")
    return figure
