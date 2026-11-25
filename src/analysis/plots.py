"""Plot aggregate results from the readmission analysis."""

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import pandas as pd


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
