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

import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from access_log_generator import generate_access_logs
from feature_engineering import engineer_features

from feature_engineering import FEATURE_COLS      # the ONE place the model's feature list is defined

HOLDOUT_SEED = 999   # different from the training seed (42)


def build_holdout_set(staff_df, patients_df):
    """Generates a fresh 30-day access log using a different random seed
    and a later time window, so it shares no rows with the training data."""
    holdout_logs = generate_access_logs(
        patients_df, staff_df,
        n_days=30, events_per_day=240,
        seed=HOLDOUT_SEED,
    )
    featured, _ = engineer_features(holdout_logs, staff_df)
    return featured


def evaluate_on_holdout(model, holdout_df):
    X = holdout_df[FEATURE_COLS].fillna(0)
    raw_scores = model.decision_function(X)          # model is NOT refit here
    predictions = model.predict(X)

    holdout_df = holdout_df.copy()
    holdout_df["anomaly_score"] = -raw_scores
    holdout_df["predicted_anomaly"] = (predictions == -1).astype(int)

    print("=== Held-Out Set Confusion Matrix ===")
    cm = confusion_matrix(holdout_df["is_anomaly"], holdout_df["predicted_anomaly"])
    print("             Predicted Normal  Predicted Anomaly")
    print(f"Actual Normal      {cm[0][0]:>10}       {cm[0][1]:>10}")
    print(f"Actual Anomaly     {cm[1][0]:>10}       {cm[1][1]:>10}")

    print("\n=== Held-Out Set Classification Report ===")
    print(classification_report(holdout_df["is_anomaly"], holdout_df["predicted_anomaly"],
                                 target_names=["Normal", "Anomaly"]))

    auc = roc_auc_score(holdout_df["is_anomaly"], holdout_df["anomaly_score"])
    print(f"Held-Out ROC-AUC: {auc:.4f}")

    return holdout_df, auc


if __name__ == "__main__":
    staff_df = pd.read_csv("data/staff.csv")
    patients_df = pd.read_csv("data/patients_with_wards.csv")
    model = joblib.load("models/isolation_forest.pkl")   # loads the EXISTING trained model

    print(f"Generating independent held-out set (seed={HOLDOUT_SEED}, not used in training)...")
    holdout_df = build_holdout_set(staff_df, patients_df)
    print(f"Held-out set size: {len(holdout_df)} events, "
          f"{holdout_df['is_anomaly'].sum()} true anomalies\n")

    holdout_df, holdout_auc = evaluate_on_holdout(model, holdout_df)

    # Compare against the training-set ROC-AUC for reference
    train_scored = pd.read_csv("data/access_logs_scored.csv")
    train_auc = roc_auc_score(train_scored["is_anomaly"], train_scored["anomaly_score"])

    print(f"\n=== Comparison ===")
    print(f"Training-set ROC-AUC:  {train_auc:.4f}")
    print(f"Held-out-set ROC-AUC:  {holdout_auc:.4f}")
    print(f"Difference:            {abs(train_auc - holdout_auc):.4f}")

    holdout_df.to_csv("data/holdout_scored.csv", index=False)
    print("\nHeld-out results saved -> data/holdout_scored.csv")
