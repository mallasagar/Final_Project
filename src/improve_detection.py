"""
improve_detection.py
How far can detection and F1 be raised, and what does it take?

  A. Current model, high tier only (score > 0).
  B. Same model, cut-off TUNED on separate validation datasets: maximise F1 subject to detection >= 90% overall AND in
     the cross-ward scenario. No new information, no retraining; only the cut-off changes.
  C. Reference: Basic ZTA rule that also enforces a ward policy (hour window OR ward mismatch).
  D. Enhanced + ROSTER CONTEXT: a staff roster tells the system who is legitimately covering another ward, so the model
     gets an "authorised ward" feature (own ward OR rostered cover). The share of legitimate cover that is on the roster
     is an ASSUMPTION (the rest is ad-hoc or emergency cover); 50%, 80% and 95% are shown. Each is compared with a
     hand-written Basic rule that uses the same roster.

Validation seeds 7100-7104 choose the cut-off; the 20 test datasets (seeds 7200-7219, 30 days each) are only used to report.
Run from the project root, after anomaly_model.py:   py src/improve_detection.py
Saves results/improve_detection.csv and results/figures/fig13_detection_improvement.png / .pdf
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
from sklearn.ensemble import IsolationForest

from access_log_generator import generate_access_logs, EVENTS_PER_DAY
from feature_engineering import engineer_features, FEATURE_COLS

VAL_SEEDS, TEST_SEEDS = range(7100, 7105), range(7200, 7220)
ROSTER_SHARES = [0.5, 0.8, 0.95]
ROSTER_FEATURES = ["hours_beyond_shift", "authorised_ward", "hour_sin", "hour_cos"]
MIN_DETECTION = 0.90


def make(seed, staff, patients):
    return engineer_features(generate_access_logs(patients, staff, n_days=30, events_per_day=EVENTS_PER_DAY, seed=seed), staff)[0]


def add_roster(d, seed, share):
    """Mark which legitimate cover events are on a roster. Anomalies are never rostered."""
    d = d.copy(); rng = np.random.default_rng(seed + 991)
    cover = (d["is_anomaly"] == 0) & (d["ward_match"] == 0)
    d["rostered"] = (cover & (rng.random(len(d)) < share)).astype(int)
    d["authorised_ward"] = ((d["ward_match"] == 1) | (d["rostered"] == 1)).astype(int)
    return d


def metrics(d, flag):
    y, ct = d["is_anomaly"].to_numpy(), d["anomaly_type"].to_numpy(); tp, fp = (flag & (y == 1)).sum(), (flag & (y == 0)).sum()
    p = tp / max(tp + fp, 1); r = tp / y.sum()
    return {"detection_%": 100 * r, "off_hours_%": 100 * flag[ct == "off_hours"].mean(), "cross_ward_%": 100 * flag[ct == "cross_ward"].mean(),
            "false_pos_%": 100 * fp / (y == 0).sum(), "precision_%": 100 * p, "F1_%": 100 * 2 * p * r / max(p + r, 1e-9)}


def best_threshold(S, y, ct):
    """Cut-off with the highest F1 among those reaching the detection target overall and in the cross-ward scenario."""
    best = None
    for t in np.unique(S):
        f = S >= t; tp, fp = (f & (y == 1)).sum(), (f & (y == 0)).sum(); r = tp / y.sum(); cw = f[ct == "cross_ward"].mean()
        p = tp / max(tp + fp, 1); f1 = 2 * p * r / max(p + r, 1e-9)
        if r >= MIN_DETECTION and cw >= MIN_DETECTION and (best is None or f1 > best[0]): best = (f1, t)
    return best[1] if best else np.unique(S)[0]


def summarise(name, kind, ms):
    df = pd.DataFrame(ms); row = {"option": name, "type": kind}
    for c in df.columns: row[c] = df[c].mean(); row[c.replace("_%", "") + "_sd"] = df[c].std(ddof=1)
    return row


if __name__ == "__main__":
    staff, patients = pd.read_csv("data/staff.csv"), pd.read_csv("data/patients_with_wards.csv")
    model = joblib.load("models/isolation_forest.pkl"); train = pd.read_csv("data/access_logs_features.csv")
    val = [(make(s, staff, patients), s) for s in VAL_SEEDS]; test = [(make(s, staff, patients), s) for s in TEST_SEEDS]
    rows = []

    score = lambda d: -model.decision_function(d[FEATURE_COLS])
    rows.append(summarise("A. Current: high tier only", "model", [metrics(d, score(d) > 0) for d, _ in test]))
    S = np.concatenate([score(d) for d, _ in val]); y = np.concatenate([d["is_anomaly"].to_numpy() for d, _ in val]); ct = np.concatenate([d["anomaly_type"].to_numpy() for d, _ in val])
    t = best_threshold(S, y, ct)
    rows.append(summarise("B. Tuned cut-off (F1-optimal, detection >= 90%)", "model", [metrics(d, score(d) >= t) for d, _ in test]))
    window = lambda d: ((d["hour"] < 7) | (d["hour"] > 20)).to_numpy()
    rows.append(summarise("C. Basic rule + ward policy (no roster)", "rule", [metrics(d, window(d) | (d["ward_match"] == 0).to_numpy()) for d, _ in test]))
    for share in ROSTER_SHARES:
        tr = add_roster(train, 42, share); m = IsolationForest(n_estimators=200, contamination=0.05, random_state=42).fit(tr[ROSTER_FEATURES])
        sc = lambda d, m=m: -m.decision_function(d[ROSTER_FEATURES])
        V = [add_roster(d, s, share) for d, s in val]; T = [add_roster(d, s, share) for d, s in test]
        S = np.concatenate([sc(d) for d in V]); y = np.concatenate([d["is_anomaly"].to_numpy() for d in V]); ct = np.concatenate([d["anomaly_type"].to_numpy() for d in V])
        t = best_threshold(S, y, ct); pct = int(share * 100)
        rows.append(summarise(f"D{pct}. Enhanced + roster ({pct}% of cover rostered)", "model", [metrics(d, sc(d) >= t) for d in T]))
        rows.append(summarise(f"R{pct}. Basic rule + same roster ({pct}%)", "rule", [metrics(d, window(d) | (d["authorised_ward"] == 0).to_numpy()) for d in T]))

    res = pd.DataFrame(rows); os.makedirs("results/figures", exist_ok=True); res.round(3).to_csv("results/improve_detection.csv", index=False)
    cols = ["detection_%", "off_hours_%", "cross_ward_%", "false_pos_%", "precision_%", "F1_%"]
    print(f"Mean over {len(TEST_SEEDS)} new 30-day datasets; the cut-off was chosen on {len(VAL_SEEDS)} separate validation datasets.\n")
    show = res[["option"] + cols].copy(); show[cols] = show[cols].round(1); print(show.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6)); labels = [o.split(". ")[0] for o in res["option"]]; x = np.arange(len(res))
    colour = ["#999999" if k == "rule" else "#D55E00" for k in res["type"]]
    for ax, cs, title in [(axes[0], [("detection_%", "Overall detection"), ("cross_ward_%", "Cross-ward detection")], "Detection (%)"), (axes[1], [("F1_%", "F1"), ("precision_%", "Precision")], "F1 and precision (%)")]:
        w = 0.38
        for j, (c, lab) in enumerate(cs):
            ax.bar(x + (j - .5) * w, res[c], w, color=colour, alpha=1 if j == 0 else .55, edgecolor="black", linewidth=.4, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.set_ylim(0, 128); ax.set_title(title, fontsize=11)
        if title.startswith("Detection"): ax.axhline(90, color="black", ls="--", lw=1); ax.text(-0.45, 92, "90% target", ha="left", fontsize=8.5)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="#777777", label=cs[0][1]), Patch(facecolor="#777777", alpha=.5, label=cs[1][1])], loc="upper center", ncol=2, fontsize=8.5, framealpha=.9)
    axes[0].set_ylabel("%"); fig.suptitle("Raising detection and F1: cut-off tuning, then roster context (orange = model, grey = hand-written rule)", fontsize=11)
    fig.text(0.5, 0.012, "A current (high tier)   B tuned cut-off   C Basic rule + ward policy   D50/D80/D95 Enhanced + roster (50/80/95% of cover rostered)   R50/R80/R95 Basic rule using the same roster", ha="center", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94)); fig.savefig("results/figures/fig13_detection_improvement.png", dpi=300); fig.savefig("results/figures/fig13_detection_improvement.pdf")
    print("\nSaved -> results/improve_detection.csv and results/figures/fig13_detection_improvement.png")
