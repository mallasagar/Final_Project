"""
hyperparameter_sensitivity.py
Does the Isolation Forest's configuration matter? Changes ONE setting at a time (others at the thesis values:
200 trees, contamination 0.05, max_samples 'auto', max_features 1.0), trains on the training log, and tests on
six brand-new 30-day datasets (seeds 6000-6005).

Run from the project root, after feature_engineering.py:
    py src/hyperparameter_sensitivity.py
Saves results/hyperparameter_sensitivity.csv
"""
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

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from access_log_generator import generate_access_logs, EVENTS_PER_DAY
from feature_engineering import engineer_features, FEATURE_COLS

TEST_SEEDS = range(6000, 6006)
GRID = [("contamination", [0.01, 0.02, 0.05, 0.10]), ("n_estimators", [100, 200, 300, 500]),
        ("max_samples", ["auto", 512, 1.0]), ("max_features", [1.0, 0.75, 0.5])]


def main():
    staff, patients = pd.read_csv("data/staff.csv"), pd.read_csv("data/patients_with_wards.csv")
    train = pd.read_csv("data/access_logs_features.csv")
    tests = [engineer_features(generate_access_logs(patients, staff, n_days=30, events_per_day=EVENTS_PER_DAY, seed=s), staff)[0] for s in TEST_SEEDS]
    rows = []
    for setting, values in GRID:
        for v in values:
            kw = dict(n_estimators=200, contamination=0.05, random_state=42); kw[setting] = v
            m = IsolationForest(**kw).fit(train[FEATURE_COLS]); res = []
            for d in tests:
                y, ct = d["is_anomaly"].to_numpy(), d["anomaly_type"].to_numpy(); f = m.predict(d[FEATURE_COLS]) == -1
                tp, fp = (f & (y == 1)).sum(), (f & (y == 0)).sum(); p = tp / max(tp + fp, 1); r = tp / y.sum()
                res.append([100 * r, 100 * f[ct == "cross_ward"].mean(), 100 * fp / (y == 0).sum(), 100 * 2 * p * r / max(p + r, 1e-9),
                            roc_auc_score(y, -m.decision_function(d[FEATURE_COLS]))])
            o = np.mean(res, axis=0)
            rows.append({"setting": setting, "value": v, "detection_%": o[0], "cross_ward_%": o[1], "false_pos_%": o[2], "F1_%": o[3], "roc_auc": o[4]})
    out = pd.DataFrame(rows); out.round(4).to_csv("results/hyperparameter_sensitivity.csv", index=False)
    print(out.round(3).to_string(index=False)); print("\nSaved -> results/hyperparameter_sensitivity.csv")


if __name__ == "__main__":
    main()
