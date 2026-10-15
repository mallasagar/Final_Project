"""
make_summary_figure.py
One-page results summary for a supervisor: four panels plus the headline metrics table, all from the
independent-datasets experiment. Saves results/figures/summary_for_supervisor.png and .pdf

Needs run_experiment_independent.py to have run. Run from the project root:
    py src/make_summary_figure.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ORDER = ["Traditional", "Basic ZTA", "Enhanced ZTA"]
COL = {"Traditional": "#999999", "Basic ZTA": "#0072B2", "Enhanced ZTA": "#D55E00"}      # colour-blind safe
HATCH = {"Traditional": "///", "Basic ZTA": "///", "Enhanced ZTA": "..."}


def human(s):
    if s >= 86400: return f"{s / 86400:.1f} days"
    if s >= 3600: return f"{s / 3600:.1f} h"
    return f"{s:.1f} s"


res = pd.read_csv("results/experiment_results_independent.csv")
cls = pd.read_csv("results/classification_metrics_independent.csv")
sc = res.groupby(["scenario", "condition"])["detection_rate"].agg(["mean", "std"])
cm = cls.groupby("condition")[["accuracy_%", "precision_%", "recall_%", "F1_%", "false_positive_%"]].agg(["mean", "std"])
tm = res.groupby("condition")[["MTTD", "MTTR", "records_exposed", "records_exposed_incl_missed"]].mean()
n_sets = res.trial.nunique()

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2], left=0.055, right=0.985, top=0.86, bottom=0.10, hspace=0.42, wspace=0.28)
fig.suptitle("Enhanced Zero Trust vs Basic Zero Trust vs Traditional security", fontsize=18, fontweight="bold", y=0.975)
fig.text(0.5, 0.925, f"Results on {n_sets} independent synthetic 30-day access logs (Isolation Forest trained once, never retrained)",
         ha="center", fontsize=11.5, color="#444444")

# ---- A: detection by scenario ----
ax = fig.add_subplot(gs[0, 0]); w = 0.26
for i, c in enumerate(ORDER):
    m = [sc.loc[("off_hours", c), "mean"], sc.loc[("cross_ward", c), "mean"]]; e = [sc.loc[("off_hours", c), "std"], sc.loc[("cross_ward", c), "std"]]
    x = np.arange(2) + (i - 1) * w
    ax.bar(x, m, w, yerr=e, capsize=3, color=COL[c], label=c)
    for xi, mi in zip(x, m): ax.text(xi, mi + 2.5, f"{mi:.0f}%", ha="center", fontsize=9)
ax.set_xticks(range(2)); ax.set_xticklabels(["Off-hours access", "Cross-ward access"], fontsize=10)
ax.set_ylim(0, 115); ax.set_ylabel("Detection rate (%)"); ax.set_title("A. Which anomalies each condition catches", fontsize=11.5, loc="left", fontweight="bold")
ax.legend(fontsize=8.5, loc="upper right")

# ---- B: detection vs false alarms ----
ax = fig.add_subplot(gs[0, 1])
for c in ORDER:
    x, y = cm.loc[c, ("false_positive_%", "mean")], cm.loc[c, ("recall_%", "mean")]
    ax.errorbar(x, y, xerr=cm.loc[c, ("false_positive_%", "std")], yerr=cm.loc[c, ("recall_%", "std")], fmt="o", ms=11, color=COL[c], capsize=3)
    ax.annotate(c, (x, y), textcoords="offset points", xytext=(10, -4 if c != "Basic ZTA" else 8), fontsize=10, color=COL[c], fontweight="bold")
ax.set_xlim(-0.25, 3.0); ax.set_ylim(0, 80)
ax.set_xlabel("False-positive rate (% of legitimate events flagged)"); ax.set_ylabel("Detection rate (% of all anomalies)")
ax.set_title("B. Detection against false alarms", fontsize=11.5, loc="left", fontweight="bold")
ax.annotate("more caught,\nsome false alarms", xy=(1.64, 61.4), xytext=(1.9, 28), fontsize=9, color="#555555", arrowprops=dict(arrowstyle="->", color="#999999"))

# ---- C: records exposed ----
ax = fig.add_subplot(gs[1, 0]); w = 0.36; x = np.arange(3)
b1 = ax.bar(x - w / 2, tm.loc[ORDER, "records_exposed"], w, color="#56B4E9", label="Detected anomalies only")
b2 = ax.bar(x + w / 2, tm.loc[ORDER, "records_exposed_incl_missed"], w, color="#E69F00", label="All anomalies, incl. missed")
for b in list(b1) + list(b2): ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, f"{b.get_height():.0f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(ORDER, fontsize=10); ax.set_ylabel("Records exposed before containment"); ax.set_ylim(0, 235)
ax.set_title("C. Patient records exposed (modelled)", fontsize=11.5, loc="left", fontweight="bold"); ax.legend(fontsize=8.5, loc="upper right")

# ---- D: timings ----
ax = fig.add_subplot(gs[1, 1]); w = 0.36
for i, c in enumerate(ORDER):
    for j, (col, alpha, lab) in enumerate([("MTTD", 1.0, "detect"), ("MTTR", 0.55, "respond")]):
        v = tm.loc[c, col]; xpos = i + (j - 0.5) * w
        ax.bar(xpos, v, w, color=COL[c], alpha=alpha, hatch=HATCH[c], edgecolor="white")
        ax.text(xpos, v * 1.35, human(v), ha="center", fontsize=8.5)
ax.set_yscale("log"); ax.set_ylim(0.3, 3e7); ax.set_xticks(range(3)); ax.set_xticklabels(ORDER, fontsize=10)
ax.set_ylabel("Seconds (log scale)"); ax.set_title("D. Time to detect (dark) and respond (light)", fontsize=11.5, loc="left", fontweight="bold")
ax.legend(handles=[Patch(facecolor="grey", hatch="///", edgecolor="white", label="Assumed range"),
                   Patch(facecolor="grey", hatch="...", edgecolor="white", label="Modelled latency")], fontsize=8.5, loc="upper right")

# ---- E: headline table ----
ax = fig.add_subplot(gs[:, 2]); ax.axis("off")
ax.set_title("Headline metrics (mean over the datasets)", fontsize=11.5, loc="left", fontweight="bold")
def pc(v): return f"{v:.1f}%"
rows = [("Detection: off-hours", [sc.loc[("off_hours", c), "mean"] for c in ORDER], pc, 1),
        ("Detection: cross-ward", [sc.loc[("cross_ward", c), "mean"] for c in ORDER], pc, 1),
        ("Detection: overall", [cm.loc[c, ("recall_%", "mean")] for c in ORDER], pc, 1),
        ("False-positive rate", [cm.loc[c, ("false_positive_%", "mean")] for c in ORDER], pc, -1),
        ("Precision", [cm.loc[c, ("precision_%", "mean")] for c in ORDER], pc, 1),
        ("F1 score", [cm.loc[c, ("F1_%", "mean")] for c in ORDER], pc, 1),
        ("Accuracy*", [cm.loc[c, ("accuracy_%", "mean")] for c in ORDER], pc, 1),
        ("Time to detect (MTTD)\u2020", [tm.loc[c, "MTTD"] for c in ORDER], human, -1),
        ("Time to respond (MTTR)\u2020", [tm.loc[c, "MTTR"] for c in ORDER], human, -1),
        ("Records exposed (detected)\u2020", [tm.loc[c, "records_exposed"] for c in ORDER], lambda v: f"{v:.0f}", -1),
        ("Records exposed (incl. missed)\u2020", [tm.loc[c, "records_exposed_incl_missed"] for c in ORDER], lambda v: f"{v:.0f}", -1)]
cells, colours = [], []
for name, vals, fmt, direction in rows:
    best = max(vals) if direction == 1 else min(vals)
    cells.append([fmt(v) for v in vals]); colours.append(["#CDEFD9" if abs(v - best) < 1e-9 else "white" for v in vals])
tbl = ax.table(cellText=cells, rowLabels=[r[0] for r in rows], colLabels=ORDER, cellColours=colours, colColours=["#EEEEEE"] * 3,
               loc="upper center", cellLoc="center", bbox=[0.43, 0.20, 0.57, 0.76])
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    if c == -1: cell.set_text_props(ha="right", fontsize=9.5); cell.set_edgecolor("none")
ax.text(0.0, 0.14, "Green = best value in the row. Overall detection pools both scenarios.\n* Accuracy is high for every condition: a system that\n   flags nothing scores 94.9%.\n\u2020 Traditional and Basic ZTA timings are assumed ranges;\n   Enhanced ZTA's are modelled latencies. Not measurements.",
        fontsize=8.8, color="#444444", va="top", transform=ax.transAxes)

fig.text(0.055, 0.035, "Synthetic data (Synthea patients, simulated access log with injected off-hours and cross-ward anomalies). "
         "Results show what the design achieves under the stated assumptions, not performance in a real hospital.", fontsize=9, color="#555555")
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/summary_for_supervisor.png", dpi=200); fig.savefig("results/figures/summary_for_supervisor.pdf")
print("saved results/figures/summary_for_supervisor.png and .pdf")
