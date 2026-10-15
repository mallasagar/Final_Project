"""
make_figures.py
Creates the nine figures recommended for the results chapter, as PNG (300 dpi) and PDF, in results/figures/.

Needs (run these first): anomaly_model.py (trained model), run_experiment_independent.py (results files).
Run from the project root:
    py src/make_figures.py
"""

import sys
from pathlib import Path


def _add_script_folders():
    here = Path(__file__).resolve().parent
    skip = {"venv", "__pycache__", "data", "results", "models"}
    for base in (here, here.parent):
        for p in [base] + [d for d in base.iterdir() if d.is_dir() and d.name not in skip and not d.name.startswith(".")]:
            if str(p) not in sys.path:
                sys.path.append(str(p))


_add_script_folders()

import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve

from access_log_generator import generate_access_logs, EVENTS_PER_DAY
from feature_engineering import engineer_features, FEATURE_COLS
from orchestration import HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD
from run_experiment import DEFAULT_WINDOW, basic_policy_flag

OUT = "results/figures"
ORDER = ["Traditional", "Basic ZTA", "Enhanced ZTA"]
COLOURS = {"Traditional": "#999999", "Basic ZTA": "#0072B2", "Enhanced ZTA": "#D55E00"}   # colour-blind safe
sns.set_theme(style="whitegrid", font_scale=1.0)


def save(fig, name, tight=True):
    os.makedirs(OUT, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=300)
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)
    print(f"saved {OUT}/{name}.png / .pdf")


def human_time(s):
    if s >= 86400: return f"{s / 86400:.1f} days"
    if s >= 3600: return f"{s / 3600:.1f} h"
    if s >= 60: return f"{s / 60:.1f} min"
    return f"{s:.1f} s"


def build_test_data(staff, patients, model, seeds=range(1000, 1005)):
    frames = []
    for s in seeds:
        d, _ = engineer_features(generate_access_logs(patients, staff, n_days=30, events_per_day=EVENTS_PER_DAY, seed=s), staff)
        d["anomaly_score"] = -model.decision_function(d[FEATURE_COLS])
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------------------------
def fig1_detection_by_scenario(res):
    g = res.groupby(["scenario", "condition"])["detection_rate"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.5)); width = 0.26
    for i, c in enumerate(ORDER):
        sub = g[g.condition == c].set_index("scenario").loc[["off_hours", "cross_ward"]]
        x = np.arange(2) + (i - 1) * width
        ax.bar(x, sub["mean"], width, yerr=sub["std"], capsize=3, color=COLOURS[c], label=c)
        for xi, m in zip(x, sub["mean"]):
            ax.text(xi, m + 3, f"{m:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(range(2)); ax.set_xticklabels(["Off-hours access", "Cross-ward access"])
    ax.set_ylabel("Detection rate (% of anomalies caught)"); ax.set_ylim(0, 112)
    ax.set_title("Detection rate by scenario (mean ± SD over 20 independent datasets)")
    ax.legend(loc="upper right")
    save(fig, "fig1_detection_by_scenario")


def fig2_operating_points(test, model_scores_col="anomaly_score"):
    y = test["is_anomaly"].values; s = test[model_scores_col].values
    fpr, tpr, _ = roc_curve(y, s)
    rule = ((test["hour"] < DEFAULT_WINDOW[0]) | (test["hour"] > DEFAULT_WINDOW[1])).values          # Traditional: hour rule only
    basic = basic_policy_flag(test).values.astype(bool)                                              # Basic ZTA: its policy (hour + roster)
    pts = {"Traditional": (100 * rule[y == 0].mean() * 0.3, 100 * 0.3 * rule[y == 1].mean()),
           "Basic ZTA": (100 * basic[y == 0].mean(), 100 * basic[y == 1].mean())}
    hi, anyt = s > HIGH_RISK_THRESHOLD, s > LOW_RISK_THRESHOLD
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(100 * fpr, 100 * tpr, color=COLOURS["Enhanced ZTA"], lw=2, alpha=.55, label="Enhanced ZTA (all possible cut-offs)")
    ax.scatter(*pts["Traditional"], s=90, color=COLOURS["Traditional"], zorder=5, label="Traditional (rule + audits)")
    ax.scatter(*pts["Basic ZTA"], s=90, color=COLOURS["Basic ZTA"], zorder=5, label="Basic ZTA (policy)")
    ax.scatter(100 * hi[y == 0].mean(), 100 * hi[y == 1].mean(), s=110, marker="D", color=COLOURS["Enhanced ZTA"], zorder=6,
               label="Enhanced ZTA: high tier only")
    ax.scatter(100 * anyt[y == 0].mean(), 100 * anyt[y == 1].mean(), s=110, marker="s", color=COLOURS["Enhanced ZTA"], edgecolor="black", zorder=6,
               label="Enhanced ZTA: medium + high tier")
    ax.set_xlim(-0.3, 15); ax.set_ylim(0, 105)
    ax.set_xlabel("False-positive rate (% of legitimate events flagged)"); ax.set_ylabel("Detection rate (% of anomalies caught)")
    ax.set_title("Detection against false alarms: where each condition operates")
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, "fig2_operating_points")


def fig3_timings(res):
    m = res.groupby("condition")[["MTTD", "MTTR"]].mean().reindex(ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    for ax, col, title in zip(axes, ["MTTD", "MTTR"], ["Mean time to detect", "Mean time to respond"]):
        for i, c in enumerate(ORDER):
            hatch = "///" if c != "Enhanced ZTA" else "..."
            ax.bar(i, m.loc[c, col], color=COLOURS[c], hatch=hatch, edgecolor="white")
            ax.text(i, m.loc[c, col] * 1.25, human_time(m.loc[c, col]), ha="center", fontsize=9)
        ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels(ORDER)
        ax.set_ylabel("Seconds (log scale)"); ax.set_title(title); ax.set_ylim(0.3, m[col].max() * 12)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="grey", hatch="///", edgecolor="white", label="Assumed range (Traditional, Basic ZTA)"),
                        Patch(facecolor="grey", hatch="...", edgecolor="white", label="Modelled latency (Enhanced ZTA)")],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig3_timings", tight=False)


def fig4_records_exposed(res):
    m = res.groupby("condition")[["records_exposed", "records_exposed_incl_missed"]].mean().reindex(ORDER)
    fig, ax = plt.subplots(figsize=(7.5, 4.5)); w = 0.36; x = np.arange(3)
    b1 = ax.bar(x - w / 2, m["records_exposed"], w, color="#56B4E9", label="Detected anomalies only")
    b2 = ax.bar(x + w / 2, m["records_exposed_incl_missed"], w, color="#E69F00", label="All anomalies (missed ones found at next manual audit)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, f"{b.get_height():.0f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(ORDER); ax.set_ylabel("Records exposed before containment")
    ax.set_title("Records exposed: the effect of missed anomalies"); ax.legend(fontsize=9)
    save(fig, "fig4_records_exposed")


def fig5_score_distributions(test):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = {"none": "Legitimate", "off_hours": "Off-hours anomaly", "cross_ward": "Cross-ward anomaly"}
    colours = {"none": "#999999", "off_hours": "#D55E00", "cross_ward": "#0072B2"}
    for k in ["none", "off_hours", "cross_ward"]:
        ax.hist(test.loc[test.anomaly_type == k, "anomaly_score"], bins=70, range=(-0.22, 0.15), density=True,
                histtype="stepfilled", alpha=.45, color=colours[k], label=labels[k])
    ax.axvline(HIGH_RISK_THRESHOLD, color="black", ls="--", lw=1); ax.axvline(LOW_RISK_THRESHOLD, color="black", ls=":", lw=1)
    ax.set_yscale("log")
    ax.text(HIGH_RISK_THRESHOLD + .003, 0.95, "high tier →", fontsize=8.5, transform=ax.get_xaxis_transform())
    ax.text(LOW_RISK_THRESHOLD + .003, 0.86, "medium tier →", fontsize=8.5, transform=ax.get_xaxis_transform())
    ax.set_xlabel("Anomaly score (higher = more unusual)"); ax.set_ylabel("Density (log scale)")
    ax.set_title("How the model scores each type of event"); ax.legend(fontsize=9)
    save(fig, "fig5_score_distributions")


def fig6_contamination(train, test):
    levels = [0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
    y = test["is_anomaly"].values; det, fp = [], []
    for c in levels:
        m = IsolationForest(n_estimators=200, contamination=c, random_state=42).fit(train[FEATURE_COLS])
        f = m.predict(test[FEATURE_COLS]) == -1
        det.append(100 * f[y == 1].mean()); fp.append(100 * f[y == 0].mean())
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot([100 * l for l in levels], det, "o-", color="#D55E00", label="Detection rate")
    ax.plot([100 * l for l in levels], fp, "s-", color="#0072B2", label="False-positive rate")
    ax.axvline(5, color="black", ls="--", lw=1); ax.text(5.2, 22, "setting used (5%) =\ntrue anomaly rate", fontsize=8.5)
    ax.set_xlabel("Assumed anomaly rate given to the model (contamination, %)"); ax.set_ylabel("%")
    ax.set_title("Sensitivity: what if the true anomaly rate is not known?"); ax.legend()
    save(fig, "fig6_contamination_sensitivity")


def _flags(test):
    """Per-event flags for the three conditions on the pooled independent test data."""
    rule = ((test["hour"] < DEFAULT_WINDOW[0]) | (test["hour"] > DEFAULT_WINDOW[1])).values
    rng = np.random.default_rng(7)
    return {"Traditional": rule & (rng.random(len(test)) < 0.30), "Basic ZTA": basic_policy_flag(test).values.astype(bool),
            "Enhanced ZTA": (test["anomaly_score"] > HIGH_RISK_THRESHOLD).values}


def fig7_confusion_heatmaps(test):
    y = test["is_anomaly"].values; flags = _flags(test)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, c in zip(axes, ORDER):
        f = flags[c].astype(int)
        cm = np.array([[((y == 0) & (f == 0)).sum(), ((y == 0) & (f == 1)).sum()],
                       [((y == 1) & (f == 0)).sum(), ((y == 1) & (f == 1)).sum()]])
        pct = 100 * cm / cm.sum(axis=1, keepdims=True)
        sns.heatmap(pct, annot=False, cmap="Blues", vmin=0, vmax=100, cbar=False, ax=ax, linewidths=1, linecolor="white")
        for i in range(2):
            for j in range(2):
                ax.text(j + .5, i + .5, f"{cm[i, j]:,}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                        color="white" if pct[i, j] > 55 else "black", fontsize=10)
        ax.set_xticklabels(["Not flagged", "Flagged"]); ax.set_yticklabels(["Legitimate", "Anomaly"], rotation=0)
        ax.set_title(c); ax.set_xlabel("System decision"); ax.set_ylabel("Actual" if c == ORDER[0] else "")
    fig.suptitle("Outcomes for every event (rows add to 100%)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig7_confusion_heatmaps", tight=False)


def fig8_tier_composition(test):
    s = test["anomaly_score"]
    tier = np.select([s < LOW_RISK_THRESHOLD, s <= HIGH_RISK_THRESHOLD], ["low", "medium"], default="high")
    t = pd.crosstab(tier, test["is_anomaly"]).reindex(["low", "medium", "high"])
    share = 100 * t[1] / t.sum(axis=1)
    action = {"low": "log only", "medium": "step-up authentication", "high": "revoke session + alert"}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(range(3), 100 - share, color="#999999", label="Legitimate events")
    ax.barh(range(3), share, left=100 - share, color="#D55E00", label="Real anomalies")
    for i, k in enumerate(["low", "medium", "high"]):
        ax.text(2, i, f"{t.loc[k].sum():,} events", va="center", color="white", fontsize=9)
        ax.text(98, i, f"{share[k]:.1f}% anomalous", va="center", ha="right", color="white" if share[k] > 30 else "black", fontsize=9)
    ax.set_yticks(range(3)); ax.set_yticklabels([f"{k.capitalize()} risk\n({action[k]})" for k in ["low", "medium", "high"]])
    ax.set_xlim(0, 100); ax.set_xlabel("Share of the tier's events (%)"); ax.invert_yaxis()
    ax.set_title("What each orchestration tier contains"); ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=2, frameon=False)
    save(fig, "fig8_tier_composition")


def fig9_spread_across_datasets(res):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, sc, title in zip(axes, ["off_hours", "cross_ward"], ["Off-hours access", "Cross-ward access"]):
        d = res[res.scenario == sc]
        sns.boxplot(data=d, x="condition", y="detection_rate", order=ORDER, hue="condition", hue_order=ORDER, palette=COLOURS, legend=False, ax=ax, showfliers=False, width=.5, boxprops={"alpha": .35})
        sns.stripplot(data=d, x="condition", y="detection_rate", order=ORDER, hue="condition", hue_order=ORDER, palette=COLOURS, legend=False, ax=ax, size=5, jitter=.15, edgecolor="black", linewidth=.4)
        ax.set_title(title); ax.set_xlabel(""); ax.set_ylabel("Detection rate (%)" if sc == "off_hours" else "")
    fig.suptitle("Spread of detection rate across the 20 independent datasets", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig9_spread_across_datasets", tight=False)


if __name__ == "__main__":
    res = pd.read_csv("results/experiment_results_independent.csv")
    staff, patients = pd.read_csv("data/staff.csv"), pd.read_csv("data/patients_with_wards.csv")
    model = joblib.load("models/isolation_forest.pkl")
    train = pd.read_csv("data/access_logs_features.csv")
    test = build_test_data(staff, patients, model)

    fig1_detection_by_scenario(res)
    fig2_operating_points(test)
    fig3_timings(res)
    fig4_records_exposed(res)
    fig5_score_distributions(test)
    fig6_contamination(train, test)
    fig7_confusion_heatmaps(test)
    fig8_tier_composition(test)
    fig9_spread_across_datasets(res)
