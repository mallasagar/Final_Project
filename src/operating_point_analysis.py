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

import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from access_log_generator import generate_access_logs, EVENTS_PER_DAY
from feature_engineering import engineer_features, FEATURE_COLS
from orchestration import HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD

RATIOS = [1, 5, 20, 100]                 # cost of a missed anomaly, relative to one false alarm
VAL_SEEDS, TEST_SEEDS = range(4000, 4005), range(4100, 4110)


def scored(seeds, staff, patients, model):
    ys, ss = [], []
    for s in seeds:
        d, _ = engineer_features(generate_access_logs(patients, staff, n_days=30, events_per_day=EVENTS_PER_DAY, seed=s), staff)
        ys.append(d["is_anomaly"].to_numpy()); ss.append(-model.decision_function(d[FEATURE_COLS]))
    return np.concatenate(ys), np.concatenate(ss)


def cost_curve(y, s, ratio):
    """Expected cost per 1,000 events for every possible cut-off (flag when score >= t). Returns t, alert rate, cost."""
    order = np.argsort(-s, kind="stable"); ys, ss = y[order], s[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[ss[:-1] != ss[1:], True]                     # evaluate only at the end of each group of tied scores
    n, p = len(y), int(y.sum())
    t = ss[last]; alerts = (tp + fp)[last] / n; cost = (ratio * (p - tp[last]) + fp[last]) / n * 1000
    return np.r_[np.inf, t], np.r_[0.0, alerts], np.r_[ratio * p / n * 1000, cost]     # first point = flag nothing


def outcome(y, flagged, ratio):
    miss, fa = int(((~flagged) & (y == 1)).sum()), int((flagged & (y == 0)).sum())
    return {"detection_%": 100 * (1 - miss / y.sum()), "false_pos_%": 100 * fa / (y == 0).sum(),
            "alerts_%": 100 * flagged.mean(), "cost_per_1000": (ratio * miss + fa) / len(y) * 1000}


if __name__ == "__main__":
    staff, patients = pd.read_csv("data/staff.csv"), pd.read_csv("data/patients_with_wards.csv")
    model = joblib.load("models/isolation_forest.pkl")
    yv, sv = scored(VAL_SEEDS, staff, patients, model); yt, st = scored(TEST_SEEDS, staff, patients, model)

    rows = []
    for r in RATIOS:
        t, alerts, cost = cost_curve(yv, sv, r); best = int(np.argmin(cost)); tb = t[best]
        for name, flagged in [(f"cost-optimal cut-off (score >= {tb:.3f})" if np.isfinite(tb) else "flag nothing", st >= tb),
                              ("model cut-off (high tier, score > 0)", st > HIGH_RISK_THRESHOLD),
                              ("medium + high tier (score > -0.05)", st > LOW_RISK_THRESHOLD)]:
            rows.append({"miss_costs_x_false_alarm": r, "policy": name, **outcome(yt, flagged, r)})
        rows.append({"miss_costs_x_false_alarm": r, "policy": "flag nothing", **outcome(yt, np.zeros(len(yt), bool), r)})
    res = pd.DataFrame(rows).round(2)
    os.makedirs("results/figures", exist_ok=True); res.to_csv("results/operating_point_analysis.csv", index=False)

    print(f"Test data: {len(yt):,} events, {int(yt.sum())} anomalies. Cost = expected cost per 1,000 events (a false alarm = 1 unit).\n")
    for r in RATIOS:
        print(f"--- a missed anomaly costs {r}x a false alarm ---")
        print(res[res.miss_costs_x_false_alarm == r].drop(columns="miss_costs_x_false_alarm").to_string(index=False)); print()

    fig, ax = plt.subplots(figsize=(8.2, 4.8)); colours = ["#009E73", "#0072B2", "#D55E00", "#CC79A7"]
    for r, c in zip(RATIOS, colours):
        t, alerts, cost = cost_curve(yv, sv, r); rel = cost / cost[0]; b = int(np.argmin(cost))
        ax.plot(100 * alerts, rel, color=c, label=f"miss = {r}x false alarm"); ax.scatter(100 * alerts[b], rel[b], color=c, s=60, zorder=5)
    ra = {k: 100 * (sv > v).mean() for k, v in [("model cut-off", HIGH_RISK_THRESHOLD), ("medium + high", LOW_RISK_THRESHOLD)]}
    for k, x in ra.items():
        ax.axvline(x, color="grey", ls="--", lw=1); ax.text(x + .15, 1.04, k, fontsize=8.5, color="grey")
    ax.set_xlim(0, 20); ax.set_ylim(0, 1.15); ax.axhline(1, color="black", lw=.8)
    ax.set_xlabel("Alert rate (% of events flagged)"); ax.set_ylabel("Cost relative to flagging nothing")
    ax.set_title("Where the cost is lowest (dots) for different costs of a missed anomaly"); ax.legend(fontsize=9, loc="center right")
    fig.tight_layout(); fig.savefig("results/figures/fig11_operating_point_cost.png", dpi=300); fig.savefig("results/figures/fig11_operating_point_cost.pdf")
    print("Saved -> results/operating_point_analysis.csv and results/figures/fig11_operating_point_cost.png")
