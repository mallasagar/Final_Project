"""
context_ablation.py
What does context add? The main design gives the model and Basic ZTA's policy two kinds of context beyond the clock: the ROSTER (who is
covering which ward) and the CARE TEAM (who has treated which patient). This script removes them one at a time, on the SAME data:

  Models (Isolation Forest, 200 trees, trained once on the 90-day log; flags chosen by a review budget fixed in advance, never by labels)
    M0  home ward only         features: hours beyond shift, ward_match, hour sin/cos          (no roster, no care)
    M1  + roster               ward_match replaced by authorised_ward (home OR rostered ward)
    M2  + roster + care        authorised_ward replaced by justified_access (home, rostered OR care team)      <- the main model
  Basic ZTA policies
    hour only / home-ward policy / hour + roster / hour + roster + care  (the main policy)

20 independent 30-day datasets (seeds 60000-60019). Also reported: fixed review budgets, the same alert volume as Basic's main policy, and two
sensitivity analyses on the assumptions that matter: how much legitimate access is NOT on the roster (the ad-hoc rate), and how much of it has a
care relationship behind it (the history share).

Run from the project root, after anomaly_model.py:   py src/context_ablation.py
Saves results/context_ablation_*.csv and results/figures/fig13_context_ablation.png
"""
import os
import sys
from pathlib import Path


def _add_script_folders():
    """Make every script folder importable, so each script can import the others whichever folder it is run from."""
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

M0 = ["hours_beyond_shift", "ward_match", "hour_sin", "hour_cos"]
M1 = ["hours_beyond_shift", "authorised_ward", "hour_sin", "hour_cos"]
TEST_SEEDS = list(range(60000, 60020))
SENS_SEEDS = TEST_SEEDS[:10]
ORDER = ["Basic, hour only", "Basic, home-ward policy", "Basic, hour + roster", "Basic ZTA (main: hour + roster + care)", "Enhanced M0 home ward only", "Enhanced M1 + roster", "Enhanced M2 + roster + care (main)"]
SHORT = ["Basic\nhour", "Basic\nhome ward", "Basic\n+roster", "Basic\n(main)", "Model\nM0", "Model\nM1", "Model\nM2 (main)"]


def make(staff, patients, seed, days=30, adhoc=None, history=None):
    """Generate one 30-day dataset (optionally with a different ad-hoc rate or care-history share) with all features, then restore the design settings."""
    saved = gen.ADHOC_RATE, gen.HISTORY_SHARE
    if adhoc is not None: gen.ADHOC_RATE = adhoc
    if history is not None: gen.HISTORY_SHARE = history
    try: return engineer_features(gen.generate_access_logs(patients, staff, n_days=days, events_per_day=gen.EVENTS_PER_DAY, seed=seed), staff)[0]
    finally: gen.ADHOC_RATE, gen.HISTORY_SHARE = saved


def metrics(d, flag):
    """Detection, per-scenario detection, false positives, precision and F1 for one set of flags."""
    y, ct = d["is_anomaly"].to_numpy(), d["anomaly_type"].to_numpy(); tp, fp = (flag & (y == 1)).sum(), (flag & (y == 0)).sum()
    p = tp / max(tp + fp, 1); r = tp / y.sum()
    return {"detection_%": 100 * r, "off_hours_%": 100 * flag[ct == "off_hours"].mean(), "cross_ward_%": 100 * flag[ct == "cross_ward"].mean(),
            "false_pos_%": 100 * fp / (y == 0).sum(), "precision_%": 100 * p, "F1_%": 100 * 2 * p * r / max(p + r, 1e-9)}


def policies(d):
    """The four Basic Zero Trust policies: hour only, home-ward policy, hour + roster, and the main policy (hour + roster + care)."""
    hour = ((d["hour"] < DEFAULT_WINDOW[0]) | (d["hour"] > DEFAULT_WINDOW[1])).to_numpy()
    return {"Basic, hour only": hour, "Basic, home-ward policy": hour | (d["ward_match"] == 0).to_numpy(), "Basic, hour + roster": hour | (d["authorised_ward"] == 0).to_numpy(),
            "Basic ZTA (main: hour + roster + care)": basic_policy_flag(d).to_numpy().astype(bool)}


def evaluate(d, models):
    """Score the four policies and the three models on one dataset; for the models also ROC-AUC, average precision and how well they separate cross-ward anomalies from legitimate cross-ward access."""
    rows = {k: metrics(d, v) for k, v in policies(d).items()}
    y = d["is_anomaly"].to_numpy(); cw = (d["anomaly_type"] == "cross_ward").to_numpy(); legit_x = ((d["is_anomaly"] == 0) & (d["ward_match"] == 0)).to_numpy()
    for name, (m, cols) in models.items():
        s = -m.decision_function(d[cols]); r = metrics(d, m.predict(d[cols]) == -1); sel = cw | legit_x
        r.update({"roc_auc": roc_auc_score(y, s), "avg_precision": average_precision_score(y, s), "anomaly_vs_legit_cross_ward_auc": roc_auc_score(cw[sel], s[sel])})
        rows[name] = r
    return rows


def topk(score, k):
    """Flag the k highest-scoring events: a review budget fixed in advance, with no labels used to choose the cut-off."""
    f = np.zeros(len(score), bool); f[np.argsort(-score, kind="stable")[:k]] = True; return f


def fit_models(train):
    """Train the two comparison models (no context, roster only) on the same training log as the main model."""
    fit = lambda cols: IsolationForest(n_estimators=200, contamination=0.05, random_state=42).fit(train[cols])
    return {"Enhanced M0 home ward only": (fit(M0), M0), "Enhanced M1 + roster": (fit(M1), M1)}


if __name__ == "__main__":
    staff, patients = pd.read_csv("data/staff.csv"), pd.read_csv("data/patients_with_wards.csv")
    # The main training log; the main model M2 was trained on it by anomaly_model.py, and M0 and M1 are trained here on the same data.
    train = pd.read_csv("data/access_logs_features.csv"); os.makedirs("results/figures", exist_ok=True)
    models = fit_models(train); models["Enhanced M2 + roster + care (main)"] = (joblib.load("models/isolation_forest.pkl"), FEATURE_COLS)
    print(f"Context ablation on {len(TEST_SEEDS)} independent 30-day datasets (seeds {TEST_SEEDS[0]}-{TEST_SEEDS[-1]}); M2 is the model in models/isolation_forest.pkl.\n")
    tests, rows = [], []
    for s in TEST_SEEDS:
        d = make(staff, patients, s); tests.append(d)
        for c, r in evaluate(d, models).items(): rows.append({"seed": s, "condition": c, **r})
    # Per-dataset results, then the means (the summary table).
    res = pd.DataFrame(rows); res.round(4).to_csv("results/context_ablation.csv", index=False)
    mc = ["detection_%", "off_hours_%", "cross_ward_%", "false_pos_%", "precision_%", "F1_%", "roc_auc", "avg_precision", "anomaly_vs_legit_cross_ward_auc"]
    summ = pd.DataFrame([{"condition": c, **{m: res[res.condition == c][m].mean() for m in mc if res[res.condition == c][m].notna().any()}} for c in ORDER])
    summ.round(4).to_csv("results/context_ablation_summary.csv", index=False)
    print("=== Mean over 20 datasets (%) ===\n" + summ[["condition", "detection_%", "off_hours_%", "cross_ward_%", "false_pos_%", "precision_%", "F1_%"]].round(1).to_string(index=False))
    print("\n=== Ranking quality of the three models (threshold-free) ===\n" + summ[summ.condition.str.startswith("Enhanced")][["condition", "roc_auc", "avg_precision", "anomaly_vs_legit_cross_ward_auc"]].round(3).to_string(index=False))
    print("   'anomaly_vs_legit_cross_ward_auc': can the model tell a cross-ward ANOMALY from legitimate cross-ward access? (0.5 = no)")
    g = lambda c: res[res.condition == c].set_index("seed")
    # Paired differences between models over the same 20 datasets, with 95% confidence intervals.
    for a, b, lab in [("Enhanced M1 + roster", "Enhanced M0 home ward only", "roster vs none"), ("Enhanced M2 + roster + care (main)", "Enhanced M1 + roster", "care on top of roster"), ("Enhanced M2 + roster + care (main)", "Enhanced M0 home ward only", "both vs none")]:
        for m in ["detection_%", "false_pos_%", "F1_%"]:
            dd = g(a)[m] - g(b)[m]; half = stats.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
            print(f"   {lab:<22} {m:<12} {dd.mean():+6.2f}  [95% CI {dd.mean() - half:+.2f}, {dd.mean() + half:+.2f}]   better in {int((dd < 0).sum() if m == 'false_pos_%' else (dd > 0).sum())}/{len(dd)} datasets")

    # Review-budget table: each model at 3-10% alerts, and at the same alert volume as Basic's main policy.
    brow = []
    for name, (m, cols) in models.items():
        for bdg in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
            brow.append({"model": name, "review_budget_%": 100 * bdg, **pd.DataFrame([metrics(d, topk(-m.decision_function(d[cols]), int(round(bdg * len(d))))) for d in tests]).mean().to_dict()})
        ms, vol = [], []
        for d in tests:
            pol = policies(d)["Basic ZTA (main: hour + roster + care)"]; ms.append(metrics(d, topk(-m.decision_function(d[cols]), int(pol.sum())))); vol.append(100 * pol.mean())
        brow.append({"model": name, "review_budget_%": float(np.mean(vol)), "note": "same alert volume as Basic's main policy", **pd.DataFrame(ms).mean().to_dict()})
    brow = pd.DataFrame(brow); brow.round(4).to_csv("results/context_ablation_budget.csv", index=False)
    print("\n=== Fixed review budgets (top x% of events by score; no labels used to choose the cut-off) ===")
    print(brow[brow.note.isna()][["model", "review_budget_%", "detection_%", "cross_ward_%", "false_pos_%", "precision_%", "F1_%"]].round(1).to_string(index=False))
    print("\n=== Same number of alerts as Basic's main policy (about %.1f%% of events) ===" % brow[brow.note.notna()]["review_budget_%"].mean())
    print(brow[brow.note.notna()][["model", "detection_%", "cross_ward_%", "false_pos_%", "precision_%", "F1_%"]].round(1).to_string(index=False))
    bp = summ.loc[summ.condition == "Basic ZTA (main: hour + roster + care)"]
    print("   (Basic's main policy itself: detection %.1f%%, false positives %.2f%%, F1 %.1f%%)" % (bp["detection_%"].iloc[0], bp["false_pos_%"].iloc[0], bp["F1_%"].iloc[0]))

    sens = []
    # Sensitivity: change one assumption at a time, regenerate the training log, retrain all three models, and test on 10 fresh datasets.
    for kind, levels in [("ad-hoc rate", [0.005, 0.015, 0.03, 0.06]), ("history share", [0.5, 0.8, 0.95])]:
        for lv in levels:
            saved = gen.ADHOC_RATE, gen.HISTORY_SHARE
            if kind == "ad-hoc rate": gen.ADHOC_RATE = lv
            else: gen.HISTORY_SHARE = lv
            try:
                tr = engineer_features(gen.generate_access_logs(patients, staff, n_days=90, events_per_day=gen.EVENTS_PER_DAY, seed=42), staff)[0]
                mm = fit_models(tr); mm["Enhanced M2 + roster + care (main)"] = (IsolationForest(n_estimators=200, contamination=0.05, random_state=42).fit(tr[FEATURE_COLS]), FEATURE_COLS)
                results = [evaluate(make(staff, patients, s), mm) for s in SENS_SEEDS]
            finally: gen.ADHOC_RATE, gen.HISTORY_SHARE = saved
            for cond in ["Basic, hour + roster", "Basic ZTA (main: hour + roster + care)", "Enhanced M1 + roster", "Enhanced M2 + roster + care (main)"]:
                sens.append({"assumption": kind, "level": lv, "condition": cond, **pd.DataFrame([r[cond] for r in results]).mean().to_dict()})
    sens = pd.DataFrame(sens); sens.round(4).to_csv("results/context_ablation_sensitivity.csv", index=False)
    for kind in ["ad-hoc rate", "history share"]:
        print(f"\n=== Sensitivity: F1 (%) as the {kind} changes (10 datasets per row; models at the main 5% review budget) ===")
        print(sens[sens.assumption == kind].pivot_table(index="level", columns="condition", values="F1_%").round(1)[["Basic, hour + roster", "Basic ZTA (main: hour + roster + care)", "Enhanced M1 + roster", "Enhanced M2 + roster + care (main)"]].to_string())

    # Figure 13: detection, F1 and false positives, sensitivity to the care-history share, and the budget curve.
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.8)); x = np.arange(len(ORDER)); colour = ["#DDDDDD", "#BBBBBB", "#8FA8CF", "#4C72B0", "#F5CBA7", "#F0B27A", "#D55E00"]
    for a_, cols, title in [(ax[0], [("detection_%", "Overall detection"), ("cross_ward_%", "Cross-ward detection")], "Detection (%)"), (ax[1], [("F1_%", "F1"), ("false_pos_%", "False positives")], "F1 and false positives (%)")]:
        for j, (c, lab) in enumerate(cols): a_.bar(x + (j - .5) * .38, [summ.loc[summ.condition == k, c].iloc[0] for k in ORDER], .38, color=colour, alpha=1 if j == 0 else .55, edgecolor="black", linewidth=.4)
        a_.set_xticks(x); a_.set_xticklabels(SHORT, fontsize=8); a_.set_ylim(0, 128); a_.set_title(title, fontsize=11)
        a_.legend(handles=[Patch(facecolor="#777", label=cols[0][1]), Patch(facecolor="#777", alpha=.5, label=cols[1][1])], loc="upper center", ncol=2, fontsize=8.5)
    for cond, c, mk in [("Enhanced M2 + roster + care (main)", "#D55E00", "o"), ("Enhanced M1 + roster", "#F0B27A", "s"), ("Basic ZTA (main: hour + roster + care)", "#4C72B0", "^"), ("Basic, hour + roster", "#8FA8CF", "d")]:
        g_ = sens[(sens.assumption == "history share") & (sens.condition == cond)]; ax[2].plot(100 * g_["level"], g_["F1_%"], mk + "-", color=c, label=cond.replace(" (main: hour + roster + care)", " (main)"))
    ax[2].set_xlabel("Ad-hoc access with a care relationship behind it (%)"); ax[2].set_ylabel("F1 (%)"); ax[2].set_ylim(0, 105); ax[2].set_title("F1 against the care-history share", fontsize=11); ax[2].legend(fontsize=7.5)
    for model, c, mk in [("Enhanced M2 + roster + care (main)", "#D55E00", "o"), ("Enhanced M1 + roster", "#F0B27A", "s"), ("Enhanced M0 home ward only", "#F5CBA7", "x")]:
        g_ = brow[(brow.model == model) & brow.note.isna()]; ax[3].plot(g_["review_budget_%"], g_["F1_%"], mk + "-", color=c, label=model)
    ax[3].axhline(bp["F1_%"].iloc[0], color="#4C72B0", ls="--", label="Basic ZTA (main), a fixed policy")
    ax[3].set_xlabel("Review budget (% of events the model may flag)"); ax[3].set_ylabel("F1 (%)"); ax[3].set_ylim(0, 105); ax[3].set_title("The models at different review budgets", fontsize=11); ax[3].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("What context adds (20 independent datasets; panels A-B use the main 5% review budget)", fontsize=11); fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("results/figures/fig13_context_ablation.png", dpi=300); fig.savefig("results/figures/fig13_context_ablation.pdf")
    print("\nSaved -> results/context_ablation_summary.csv, context_ablation.csv, context_ablation_budget.csv, context_ablation_sensitivity.csv, results/figures/fig13_context_ablation.png")
