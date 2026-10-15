"""
test_on_holdout_multi.py
Tests the ALREADY-TRAINED Isolation Forest on several brand-new access logs (different random seeds,
never seen in training) without retraining. If performance stays close to the training figures, the
model has learned a general pattern and not just the quirks of one dataset.

Run from the project root, after anomaly_model.py:
    py src/test_on_holdout_multi.py
"""

import sys
from pathlib import Path


def _add_script_folders():
    """Let this script find the other pipeline scripts wherever they sit (src/, src/pipeline/, src/model/)."""
    here = Path(__file__).resolve().parent
    skip = {"venv", "__pycache__", "data", "results", "models", "logs"}
    for base in (here, here.parent):
        for p in [base] + [d for d in base.iterdir() if d.is_dir() and d.name not in skip and not d.name.startswith(".")]:
            if str(p) not in sys.path:
                sys.path.append(str(p))


_add_script_folders()

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from access_log_generator import generate_access_logs, EVENTS_PER_DAY
from feature_engineering import engineer_features, FEATURE_COLS

HOLDOUT_SEEDS = [999, 888, 777, 666, 555, 444, 333]
HOLDOUT_DAYS = 30


def evaluate_seed(seed, staff_df, patients_df, model):
    logs = generate_access_logs(patients_df, staff_df, n_days=HOLDOUT_DAYS, events_per_day=EVENTS_PER_DAY, seed=seed)
    df, _ = engineer_features(logs, staff_df)
    X = df[FEATURE_COLS].fillna(0)
    flagged = (model.predict(X) == -1)
    y = df["is_anomaly"].values
    return {
        "seed": seed,
        "events": len(df),
        "anomalies": int(y.sum()),
        "roc_auc": roc_auc_score(y, -model.decision_function(X)),
        "detection_rate_%": 100 * flagged[y == 1].mean(),
        "false_positive_%": 100 * flagged[y == 0].mean(),
    }


if __name__ == "__main__":
    staff_df = pd.read_csv("data/staff.csv")
    patients_df = pd.read_csv("data/patients_with_wards.csv")
    model = joblib.load("models/isolation_forest.pkl")

    train = pd.read_csv("data/access_logs_scored.csv")
    train_row = {
        "seed": "training",
        "events": len(train),
        "anomalies": int(train["is_anomaly"].sum()),
        "roc_auc": roc_auc_score(train["is_anomaly"], train["anomaly_score"]),
        "detection_rate_%": 100 * train.loc[train["is_anomaly"] == 1, "predicted_anomaly"].mean(),
        "false_positive_%": 100 * train.loc[train["is_anomaly"] == 0, "predicted_anomaly"].mean(),
    }

    held_out = pd.DataFrame([evaluate_seed(s, staff_df, patients_df, model) for s in HOLDOUT_SEEDS])
    table = pd.concat([pd.DataFrame([train_row]), held_out], ignore_index=True)
    table.round(3).to_csv("results/holdout_summary.csv", index=False)

    print(table.round(3).to_string(index=False))
    print(f"\nHeld-out ({len(HOLDOUT_SEEDS)} sets):  ROC-AUC {held_out['roc_auc'].mean():.4f} +/- {held_out['roc_auc'].std():.4f}"
          f"   detection rate {held_out['detection_rate_%'].mean():.1f}% +/- {held_out['detection_rate_%'].std():.1f}"
          f"   false positives {held_out['false_positive_%'].mean():.1f}% +/- {held_out['false_positive_%'].std():.1f}")
    print(f"Training:               ROC-AUC {train_row['roc_auc']:.4f}   detection rate {train_row['detection_rate_%']:.1f}%"
          f"   false positives {train_row['false_positive_%']:.1f}%")
    print("\nSaved -> results/holdout_summary.csv")
