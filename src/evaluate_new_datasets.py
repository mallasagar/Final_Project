"""
evaluate_new_datasets.py
Tests the already-trained Isolation Forest on MANY brand-new datasets that this script creates itself.

Each dataset is a fresh simulated access log (its own random seed, 30 days by default). The model is never retrained.
All three security conditions are scored on every dataset, so the comparison is PAIRED: for each dataset we can ask
"did Enhanced ZTA beat Basic ZTA here?" and summarise across datasets with confidence intervals and Wilcoxon tests.

The default seeds (20000 upwards) are new: they were not used to tune or explore anything, so this is a clean
confirmation set. Do not change the model after looking at these results.

Run from the project root, after anomaly_model.py:
    py src/evaluate_new_datasets.py                 # 100 new datasets
    py src/evaluate_new_datasets.py --n 50          # a different number
    py src/evaluate_new_datasets.py --n 100 --save-data   # also write each dataset to data/new_test_sets/

Outputs (results/): new_datasets_per_dataset.csv, new_datasets_summary.csv, new_datasets_paired_tests.csv
and results/figures/fig12_new_datasets.png
"""

import argparse
import os
import sys
import warnings
from pathlib import Path


def _add_script_folders():
    here = Path(__file__).resolve().parent
    skip = {"venv", "__pycache__", "data", "results", "models", "logs"}
    for base in (here, here.parent):
        for p in [base] + [d for d in base.iterdir() if d.is_dir() and d.name not in skip and not d.name.startswith(".")]:
            if str(p) not in sys.path:
                sys.path.append(str(p))


_add_script_folders()

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import run_experiment_independent as rei
from run_experiment import simulate_condition, CONDITIONS, DEFAULT_WINDOW

ORDER = ["Traditional", "Basic ZTA", "Enhanced ZTA"]
COL = {"Traditional": "#999999", "Basic ZTA": "#0072B2", "Enhanced ZTA": "#D55E00"}
KEY_METRICS = ["detection_overall", "detection_off_hours", "detection_cross_ward", "false_positive", "precision", "f1", "accuracy",
               "records_incl_missed"]
LOWER_IS_BETTER = {"false_positive", "records_incl_missed", "records_detected", "MTTD", "MTTR"}


def dataset_rows(df, seed):
    y, ct = df["is_anomaly"].to_numpy(), df["anomaly_type"].to_numpy()
    rows = []
    for ci, cond in enumerate(CONDITIONS):
        sim = simulate_condition(df, cond, np.random.default_rng([seed, 77, ci]), DEFAULT_WINDOW)
        f = sim["detected"].to_numpy().astype(bool)
        tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / y.sum()
        caught = sim[(sim.is_anomaly == 1) & (sim.detected == 1)]
        rows.append({"seed": seed, "condition": cond,
                     "detection_overall": 100 * recall, "detection_off_hours": 100 * f[ct == "off_hours"].mean(),
                     "detection_cross_ward": 100 * f[ct == "cross_ward"].mean(), "false_positive": 100 * f[y == 0].mean(),
                     "precision": 100 * precision, "f1": 100 * 2 * precision * recall / (precision + recall) if tp else np.nan,
                     "accuracy": 100 * (f == y.astype(bool)).mean(),
                     "MTTD": caught["mttd_sec"].mean(), "MTTR": caught["mttr_sec"].mean(),
                     "records_detected": caught["records_exposed"].mean(),
                     "records_incl_missed": sim.loc[sim.is_anomaly == 1, "records_exposed_incl_missed"].mean(),
                     "roc_auc": roc_auc_score(y, df["anomaly_score"]) if cond == "Enhanced ZTA" else np.nan})
    return rows


def summarise(res):
    out = []
    for cond in ORDER:
        for m in KEY_METRICS + ["MTTD", "MTTR", "records_detected", "roc_auc"]:
            v = res.loc[res.condition == cond, m].dropna()
            if v.empty: continue
            n, mean, sd = len(v), v.mean(), v.std(ddof=1)
            half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
            out.append({"condition": cond, "metric": m, "n_datasets": n, "mean": mean, "sd": sd,
                        "ci95_low": mean - half, "ci95_high": mean + half, "min": v.min(), "max": v.max()})
    return pd.DataFrame(out)


def paired_tests(res):
    wide = {m: res.pivot(index="seed", columns="condition", values=m) for m in KEY_METRICS}
    rows = []
    for a, b in [("Enhanced ZTA", "Basic ZTA"), ("Enhanced ZTA", "Traditional"), ("Basic ZTA", "Traditional")]:
        for m in KEY_METRICS:
            d = (wide[m][a] - wide[m][b]).dropna(); n = len(d)
            if n < 2: continue
            half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = stats.wilcoxon(d).pvalue
            except ValueError:
                p = np.nan                                                # every difference is exactly zero
            better = (d < 0) if m in LOWER_IS_BETTER else (d > 0)
            rows.append({"A": a, "B": b, "metric": m, "mean_difference_A_minus_B": d.mean(), "ci95_low": d.mean() - half,
                         "ci95_high": d.mean() + half, "wilcoxon_p": p, "A_better_in_%_of_datasets": 100 * better.mean(), "n": n})
    return pd.DataFrame(rows)


def fmt_p(p):
    return "n/a" if pd.isna(p) else ("< .001" if p < .001 else f"{p:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Test the trained model on many freshly generated datasets.")
    ap.add_argument("--n", type=int, default=100, help="number of new datasets (default 100)")
    ap.add_argument("--days", type=int, default=30, help="days per dataset (default 30)")
    ap.add_argument("--start-seed", type=int, default=20000, help="first seed; use a range you have never used before")
    ap.add_argument("--save-data", action="store_true", help="write each dataset to data/new_test_sets/")
    a = ap.parse_args()

    staff, patients = pd.read_csv("data/staff.csv"), pd.read_csv("data/patients_with_wards.csv")
    model = joblib.load("models/isolation_forest.pkl")
    rei.TRIAL_DAYS = a.days
    seeds = list(range(a.start_seed, a.start_seed + a.n))
    if a.save_data: os.makedirs("data/new_test_sets", exist_ok=True)

    print(f"Creating {a.n} new datasets (seeds {seeds[0]}-{seeds[-1]}, {a.days} days each) and testing the trained model on each...")
    rows = []
    for i, s in enumerate(seeds, 1):
        df = rei.make_trial_dataset(s, staff, patients, model)
        rows += dataset_rows(df, s)
        if a.save_data:
            df[["timestamp", "user_id", "role", "assigned_ward", "patient_id", "patient_ward", "action", "is_anomaly", "anomaly_type",
                "anomaly_score", "predicted_anomaly", "risk_tier"]].to_csv(f"data/new_test_sets/set_{s}.csv", index=False)
        if i % 25 == 0 or i == a.n: print(f"  {i}/{a.n} datasets done")

    res = pd.DataFrame(rows); summ = summarise(res); paired = paired_tests(res)
    os.makedirs("results/figures", exist_ok=True)
    res.round(3).to_csv("results/new_datasets_per_dataset.csv", index=False)
    summ.round(3).to_csv("results/new_datasets_summary.csv", index=False); paired.round(4).to_csv("results/new_datasets_paired_tests.csv", index=False)

    print(f"\n=== Mean [95% confidence interval] over {a.n} new datasets ===")
    labels = {"detection_overall": "Detection: overall %", "detection_off_hours": "Detection: off-hours %", "detection_cross_ward": "Detection: cross-ward %",
              "false_positive": "False-positive rate %", "precision": "Precision %", "f1": "F1 %", "accuracy": "Accuracy %", "records_incl_missed": "Records exposed (incl. missed)"}
    tab = pd.DataFrame({c: {labels[m]: "{:.1f} [{:.1f}, {:.1f}]".format(*summ[(summ.condition == c) & (summ.metric == m)][["mean", "ci95_low", "ci95_high"]].iloc[0])
                            for m in KEY_METRICS} for c in ORDER})
    print(tab.to_string())
    au = summ[(summ.condition == "Enhanced ZTA") & (summ.metric == "roc_auc")].iloc[0]
    print(f"\nEnhanced ZTA ROC-AUC: {au['mean']:.4f} [{au['ci95_low']:.4f}, {au['ci95_high']:.4f}]   (range {au['min']:.3f}-{au['max']:.3f})")

    print("\n=== Paired comparison: is Enhanced ZTA better, dataset by dataset? ===")
    for b in ["Basic ZTA", "Traditional"]:
        print(f"\n  Enhanced ZTA minus {b}:")
        q = paired[(paired.A == "Enhanced ZTA") & (paired.B == b) & paired.metric.isin(["detection_overall", "detection_cross_ward", "false_positive", "f1", "records_incl_missed"])]
        for _, r in q.iterrows():
            print(f"    {labels[r.metric]:<32} difference {r.mean_difference_A_minus_B:+7.2f}  [{r.ci95_low:+.2f}, {r.ci95_high:+.2f}]   "
                  f"Wilcoxon p {fmt_p(r.wilcoxon_p):<8} better in {r['A_better_in_%_of_datasets']:.0f}% of datasets")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    for i, c in enumerate(ORDER):
        v = res.loc[res.condition == c, "detection_overall"]
        ax.boxplot(v, positions=[i], widths=.5, showfliers=False, patch_artist=True, boxprops=dict(facecolor=COL[c], alpha=.3), medianprops=dict(color="black"))
        ax.scatter(np.random.default_rng(1).normal(i, .06, len(v)), v, s=14, color=COL[c], edgecolor="black", linewidth=.3, zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels(ORDER); ax.set_ylabel("Overall detection rate (%)")
    ax.set_title(f"Detection rate on {a.n} new datasets", fontsize=11)
    d = (res[res.condition == "Enhanced ZTA"].set_index("seed").detection_overall - res[res.condition == "Basic ZTA"].set_index("seed").detection_overall)
    axes[1].hist(d, bins=15, color=COL["Enhanced ZTA"], alpha=.8, edgecolor="white"); axes[1].axvline(0, color="black", lw=1.2, ls="--")
    axes[1].set_xlabel("Enhanced ZTA minus Basic ZTA, overall detection (percentage points)"); axes[1].set_ylabel("Number of datasets")
    axes[1].set_title(f"Better in {100 * (d > 0).mean():.0f}% of datasets (dashed line = no difference)", fontsize=11)
    fig.tight_layout(); fig.savefig("results/figures/fig12_new_datasets.png", dpi=300); fig.savefig("results/figures/fig12_new_datasets.pdf")
    print("\nSaved: results/new_datasets_per_dataset.csv, new_datasets_summary.csv, new_datasets_paired_tests.csv, results/figures/fig12_new_datasets.png")
