"""Plot exact count distributions for the full prepared cohort."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import pandas as pd

from src.eda.descriptive import PRIOR_VISIT_COLUMNS


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
