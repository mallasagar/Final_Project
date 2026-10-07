import os
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

import access_log_generator as gen
from feature_engineering import engineer_features, FEATURE_COLS
from run_experiment import DEFAULT_WINDOW

TEST_SEEDS = [5000, 5001, 5002]
OUTSIDE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
COVER_LEVELS = [0.0, 0.025, 0.05, 0.10, 0.20]


def make(staff, patients, days, seed):
    return engineer_features(gen.generate_access_logs(patients, staff, n_days=days, events_per_day=gen.EVENTS_PER_DAY, seed=seed), staff)[0]


def run_setting(outside, cover, staff, patients):
    gen.OUTSIDE_SHIFT_RATE, gen.LEGIT_CROSS_WARD_RATE = outside, cover
    train = make(staff, patients, 90, 42)
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42).fit(train[FEATURE_COLS])
    rows = []
    for s in TEST_SEEDS:
        d = make(staff, patients, 30, s); y = d["is_anomaly"].values; ct = d["anomaly_type"].values
        flag = model.predict(d[FEATURE_COLS]) == -1
        basic = ((d["hour"] < DEFAULT_WINDOW[0]) | (d["hour"] > DEFAULT_WINDOW[1])).values
        rows.append({"enh_overall": 100 * flag[y == 1].mean(), "enh_off_hours": 100 * flag[ct == "off_hours"].mean(),
                     "enh_cross_ward": 100 * flag[ct == "cross_ward"].mean(), "enh_false_pos": 100 * flag[y == 0].mean(),
                     "enh_auc": roc_auc_score(y, -model.decision_function(d[FEATURE_COLS])),
                     "basic_overall": 100 * basic[y == 1].mean(), "basic_false_pos": 100 * basic[y == 0].mean()})
    return pd.DataFrame(rows).mean().to_dict()


if __name__ == "__main__":
    staff, patients = pd.read_csv("data/staff.csv"), pd.read_csv("data/patients_with_wards.csv")
    out = []
    for kind, levels, fixed in [("outside-shift", OUTSIDE_LEVELS, 0.05), ("cross-ward", COVER_LEVELS, 0.10)]:
        for lv in levels:
            o, c = (lv, fixed) if kind == "outside-shift" else (fixed, lv)
            out.append({"noise": kind, "level_%": 100 * lv, **run_setting(o, c, staff, patients)})
    gen.OUTSIDE_SHIFT_RATE, gen.LEGIT_CROSS_WARD_RATE = 0.10, 0.05          # restore the design values
    res = pd.DataFrame(out).round(2)
    os.makedirs("results/figures", exist_ok=True)
    res.to_csv("results/noise_sensitivity.csv", index=False)

    cols = ["level_%", "enh_overall", "enh_off_hours", "enh_cross_ward", "enh_false_pos", "enh_auc", "basic_overall", "basic_false_pos"]
    for kind, note in [("outside-shift", "(cross-ward cover fixed at 5%; design value is 10%)"), ("cross-ward", "(outside-shift fixed at 10%; design value is 5%)")]:
        print(f"\n=== Varying the {kind} noise {note} ===")
        print(res[res.noise == kind][cols].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, kind, title, design in zip(axes, ["outside-shift", "cross-ward"],
                                       ["Legitimate events one hour outside the shift", "Legitimate cross-ward cover"], [10, 5]):
        d = res[res.noise == kind]
        ax.plot(d["level_%"], d["enh_overall"], "o-", color="#D55E00", label="Enhanced ZTA detection rate")
        ax.plot(d["level_%"], d["enh_cross_ward"], "^--", color="#D55E00", alpha=.6, label="  of which cross-ward")
        ax.plot(d["level_%"], d["enh_false_pos"], "s-", color="#0072B2", label="Enhanced ZTA false positives")
        ax.plot(d["level_%"], d["basic_overall"], "d:", color="#009E73", label="Basic ZTA detection rate")
        ax.axvline(design, color="black", ls="--", lw=1); ax.text(design, 103, " design value", fontsize=8.5)
        ax.set_xlabel("Noise level (% of legitimate events)"); ax.set_title(title, fontsize=10); ax.set_ylim(-3, 112)
    axes[0].set_ylabel("%")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Sensitivity to noise (retrained and tested at each level)", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig("results/figures/fig10_noise_sensitivity.png", dpi=300); fig.savefig("results/figures/fig10_noise_sensitivity.pdf")
    print("\nSaved -> results/noise_sensitivity.csv and results/figures/fig10_noise_sensitivity.png")
