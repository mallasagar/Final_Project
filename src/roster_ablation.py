"""
roster_ablation.py
What does the roster add? The main design uses the roster (feature 'authorised_ward', and Basic ZTA's policy). This script removes
it, one piece at a time, on the SAME data, so its contribution can be seen:

  Enhanced (main)            Isolation Forest with 'authorised_ward' (home ward OR rostered ward)        <- the main model
  Enhanced, no roster        the same model, but with 'ward_match' (home ward only)
  Basic ZTA (main)           policy: hour window AND roster authorisation
  Basic, hour rule only      the weaker Basic: hour window only (cannot see cross-ward access)
  Basic, ward policy         hour window OR any access outside the HOME ward (ignores the roster)

20 independent 30-day datasets (seeds 60000-60019); both models use contamination 0.05 (a fixed 5% review budget; no label-based
tuning). Also reported: fixed review budgets, the same alert volume as Basic's policy, and a sensitivity analysis on how much
legitimate cross-ward access is NOT on the roster (the ad-hoc rate).

Run from the project root, after anomaly_model.py:   py src/roster_ablation.py
Saves results/roster_ablation_*.csv and results/figures/fig13_roster_ablation.png
"""
import os
import sys
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
from matplotlib.patches import Patch
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

import access_log_generator as gen
from feature_engineering import engineer_features, FEATURE_COLS
from run_experiment import basic_policy_flag, DEFAULT_WINDOW

NO_ROSTER = ["hours_beyond_shift", "ward_match", "hour_sin", "hour_cos"]
TEST_SEEDS = list(range(60000, 60020))
SENS_RATES, SENS_SEEDS = [0.005, 0.015, 0.03, 0.06], list(range(60000, 60010))
ORDER = ["Basic, hour rule only", "Basic, ward policy", "Basic ZTA (main)", "Enhanced, no roster", "Enhanced (main)"]


def make(staff, patients, seed, adhoc=None):
    old = gen.ADHOC_RATE
    if adhoc is not None: gen.ADHOC_RATE = adhoc
    try: return engineer_features(gen.generate_access_logs(patients, staff, n_days=30, events_per_day=gen.EVENTS_PER_DAY, seed=seed), staff)[0]
    finally: gen.ADHOC_RATE = old


def metrics(d, flag):
    y, ct = d["is_anomaly"].to_numpy(), d["anomaly_type"].to_numpy(); tp, fp = (flag & (y == 1)).sum(), (flag & (y == 0)).sum()
    p = tp / max(tp + fp, 1); r = tp / y.sum()
    return {"detection_%": 100 * r, "off_hours_%": 100 * flag[ct == "off_hours"].mean(), "cross_ward_%": 100 * flag[ct == "cross_ward"].mean(),
            "false_pos_%": 100 * fp / (y == 0).sum(), "precision_%": 100 * p, "F1_%": 100 * 2 * p * r / max(p + r, 1e-9)}


def evaluate(d, m_main, m0):
    hour = ((d["hour"] < DEFAULT_WINDOW[0]) | (d["hour"] > DEFAULT_WINDOW[1])).to_numpy()
    rows = {"Basic, hour rule only": metrics(d, hour), "Basic, ward policy": metrics(d, hour | (d["ward_match"] == 0).to_numpy()),
            "Basic ZTA (main)": metrics(d, basic_policy_flag(d).to_numpy().astype(bool))}
    y = d["is_anomaly"].to_numpy(); cw = (d["anomaly_type"] == "cross_ward").to_numpy(); legit_x = ((d["is_anomaly"] == 0) & (d["ward_match"] == 0)).to_numpy()
    for name, m, cols in [("Enhanced, no roster", m0, NO_ROSTER), ("Enhanced (main)", m_main, FEATURE_COLS)]:
        s = -m.decision_function(d[cols]); r = metrics(d, m.predict(d[cols]) == -1); sel = cw | legit_x
        r.update({"roc_auc": roc_auc_score(y, s), "avg_precision": average_precision_score(y, s), "anomaly_vs_legit_cross_ward_auc": roc_auc_score(cw[sel], s[sel])})
        rows[name] = r
    return rows


def topk(score, k):
    flag = np.zeros(len(score), bool); flag[np.argsort(-score, kind="stable")[:k]] = True; return flag


if __name__ == "__main__":
    staff, patients = pd.read_csv("data/staff.csv"), pd.read_csv("data/patients_with_wards.csv")
    train = pd.read_csv("data/access_logs_features.csv"); m_main = joblib.load("models/isolation_forest.pkl")
    m0 = IsolationForest(n_estimators=200, contamination=0.05, random_state=42).fit(train[NO_ROSTER])
    os.makedirs("results/figures", exist_ok=True)
    print(f"Roster ablation on {len(TEST_SEEDS)} independent 30-day datasets (seeds {TEST_SEEDS[0]}-{TEST_SEEDS[-1]}); the main model is the one in models/isolation_forest.pkl.\n")
    tests, rows = [], []
    for s in TEST_SEEDS:
        d = make(staff, patients, s); tests.append(d)
        for c, r in evaluate(d, m_main, m0).items(): rows.append({"seed": s, "condition": c, **r})
    res = pd.DataFrame(rows); res.round(4).to_csv("results/roster_ablation.csv", index=False)
    mc = ["detection_%", "off_hours_%", "cross_ward_%", "false_pos_%", "precision_%", "F1_%", "roc_auc", "avg_precision", "anomaly_vs_legit_cross_ward_auc"]
    summ = []
    for c in ORDER:
        g = res[res.condition == c]; row = {"condition": c}
        for m in mc:
            if m in g and g[m].notna().any(): row[m] = g[m].mean()
        summ.append(row)
    summ = pd.DataFrame(summ); summ.round(4).to_csv("results/roster_ablation_summary.csv", index=False)
    print("=== Mean over 20 datasets (%) ===\n" + summ[["condition", "detection_%", "off_hours_%", "cross_ward_%", "false_pos_%", "precision_%", "F1_%"]].round(1).to_string(index=False))
    print("\n=== Ranking quality of the two models (threshold-free) ===\n" + summ[summ.condition.str.startswith("Enhanced")][["condition", "roc_auc", "avg_precision", "anomaly_vs_legit_cross_ward_auc"]].round(3).to_string(index=False))
    print("   'anomaly_vs_legit_cross_ward_auc': can the model tell a cross-ward ANOMALY from legitimate cross-ward access? (0.5 = no)")
    a = res[res.condition == "Enhanced (main)"].set_index("seed"); b = res[res.condition == "Enhanced, no roster"].set_index("seed")
    for m in ["detection_%", "cross_ward_%", "false_pos_%", "F1_%"]:
        dd = a[m] - b[m]; half = stats.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
        print(f"   with roster minus without, {m:<13} {dd.mean():+6.2f}  [95% CI {dd.mean() - half:+.2f}, {dd.mean() + half:+.2f}]   better in {int((dd < 0).sum() if m == 'false_pos_%' else (dd > 0).sum())}/{len(dd)} datasets")

    brow = []
    for name, m, cols in [("Enhanced, no roster", m0, NO_ROSTER), ("Enhanced (main)", m_main, FEATURE_COLS)]:
        for bdg in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
            ms = [metrics(d, topk(-m.decision_function(d[cols]), int(round(bdg * len(d))))) for d in tests]
            brow.append({"model": name, "review_budget_%": 100 * bdg, **pd.DataFrame(ms).mean().to_dict()})
        ms, vol = [], []
        for d in tests:
            pol = basic_policy_flag(d).to_numpy().astype(bool); ms.append(metrics(d, topk(-m.decision_function(d[cols]), int(pol.sum())))); vol.append(100 * pol.mean())
        brow.append({"model": name, "review_budget_%": float(np.mean(vol)), "note": "same alert volume as Basic ZTA's policy", **pd.DataFrame(ms).mean().to_dict()})
    brow = pd.DataFrame(brow); brow.round(4).to_csv("results/roster_ablation_budget.csv", index=False)
    print("\n=== Fixed review budgets (top x% of events by score; no labels used to choose the cut-off) ===")
    print(brow[brow.note.isna()][["model", "review_budget_%", "detection_%", "cross_ward_%", "false_pos_%", "precision_%", "F1_%"]].round(1).to_string(index=False))
    print("\n=== Same number of alerts as Basic ZTA's policy (about %.1f%% of events) ===" % brow[brow.note.notna()]["review_budget_%"].mean())
    print(brow[brow.note.notna()][["model", "detection_%", "cross_ward_%", "false_pos_%", "precision_%", "F1_%"]].round(1).to_string(index=False))
    bp = summ.loc[summ.condition == "Basic ZTA (main)"]
    print("   (Basic ZTA's policy itself: detection %.1f%%, false positives %.1f%%, F1 %.1f%%)" % (bp["detection_%"].iloc[0], bp["false_pos_%"].iloc[0], bp["F1_%"].iloc[0]))

    sens = []
    for rate in SENS_RATES:
        old = gen.ADHOC_RATE; gen.ADHOC_RATE = rate
        try:
            tr = engineer_features(gen.generate_access_logs(patients, staff, n_days=90, events_per_day=gen.EVENTS_PER_DAY, seed=42), staff)[0]
            a1 = IsolationForest(n_estimators=200, contamination=0.05, random_state=42).fit(tr[FEATURE_COLS]); a0 = IsolationForest(n_estimators=200, contamination=0.05, random_state=42).fit(tr[NO_ROSTER])
            lm = tr[(tr.is_anomaly == 0) & (tr.ward_match == 0)]; on_roster = 100 * (lm.authorised_ward == 1).mean()
            results = [evaluate(make(staff, patients, s, rate), a1, a0) for s in SENS_SEEDS]
        finally: gen.ADHOC_RATE = old
        for cond in ["Basic, ward policy", "Basic ZTA (main)", "Enhanced, no roster", "Enhanced (main)"]:
            sens.append({"adhoc_rate": rate, "legit_cross_ward_on_roster_%": on_roster, "condition": cond, **pd.DataFrame([r[cond] for r in results]).mean().to_dict()})
    sens = pd.DataFrame(sens); sens.round(4).to_csv("results/roster_ablation_sensitivity.csv", index=False)
    print("\n=== Sensitivity: F1 (%) when more legitimate cross-ward access is NOT on the roster (10 datasets per row) ===")
    print(sens.pivot_table(index=["adhoc_rate", "legit_cross_ward_on_roster_%"], columns="condition", values="F1_%").round(1)[["Basic, ward policy", "Basic ZTA (main)", "Enhanced, no roster", "Enhanced (main)"]].to_string())

    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8)); x = np.arange(len(ORDER)); colour = ["#BBBBBB", "#777777", "#4C72B0", "#F0B27A", "#D55E00"]
    labels = ["Basic\nhour only", "Basic\n+ward", "Basic\n(main)", "Enh.\nno roster", "Enh.\n(main)"]
    for a_, cols, title in [(ax[0], [("detection_%", "Overall detection"), ("cross_ward_%", "Cross-ward detection")], "Detection (%)"), (ax[1], [("F1_%", "F1"), ("precision_%", "Precision")], "F1 and precision (%)")]:
        for j, (c, lab) in enumerate(cols): a_.bar(x + (j - .5) * .38, [summ.loc[summ.condition == k, c].iloc[0] for k in ORDER], .38, color=colour, alpha=1 if j == 0 else .55, edgecolor="black", linewidth=.4)
        a_.set_xticks(x); a_.set_xticklabels(labels, fontsize=8.5); a_.set_ylim(0, 128); a_.set_title(title, fontsize=11)
        a_.legend(handles=[Patch(facecolor="#777", label=cols[0][1]), Patch(facecolor="#777", alpha=.5, label=cols[1][1])], loc="upper center", ncol=2, fontsize=8.5)
    for cond, c, mk in [("Enhanced (main)", "#D55E00", "o"), ("Enhanced, no roster", "#F0B27A", "s"), ("Basic ZTA (main)", "#4C72B0", "^"), ("Basic, ward policy", "#777777", "d")]:
        g = sens[sens.condition == cond]; ax[2].plot(g["legit_cross_ward_on_roster_%"], g["F1_%"], mk + "-", color=c, label=cond)
    ax[2].set_xlabel("Legitimate cross-ward access that is on the roster (%)"); ax[2].set_ylabel("F1 (%)"); ax[2].set_title("F1 against roster coverage", fontsize=11); ax[2].legend(fontsize=8); ax[2].set_ylim(0, 105)
    for model, c, mk in [("Enhanced (main)", "#D55E00", "o"), ("Enhanced, no roster", "#F0B27A", "s")]:
        g = brow[(brow.model == model) & brow.note.isna()]; ax[3].plot(g["review_budget_%"], g["F1_%"], mk + "-", color=c, label=model)
    ax[3].axhline(summ.loc[summ.condition == "Basic ZTA (main)", "F1_%"].iloc[0], color="#4C72B0", ls="--", label="Basic ZTA (main), a fixed policy")
    ax[3].set_xlabel("Review budget (% of events the model may flag)"); ax[3].set_ylabel("F1 (%)"); ax[3].set_ylim(0, 105); ax[3].set_title("The model at different review budgets", fontsize=11); ax[3].legend(fontsize=8, loc="lower right")
    fig.suptitle("What the roster adds (20 independent datasets; panels A-B use the main 5% review budget)", fontsize=11); fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("results/figures/fig13_roster_ablation.png", dpi=300); fig.savefig("results/figures/fig13_roster_ablation.pdf")
    print("\nSaved -> results/roster_ablation_summary.csv, roster_ablation.csv, roster_ablation_budget.csv, roster_ablation_sensitivity.csv, results/figures/fig13_roster_ablation.png")
