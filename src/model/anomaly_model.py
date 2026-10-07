"""
anomaly_model.py
Trains an Isolation Forest model on the engineered access-log features
and evaluates how well it separates labelled anomalies from normal events.

================================================================================
HOW ISOLATION FOREST ACTUALLY TRAINS (explained inline)
================================================================================

Unlike most machine learning models, Isolation Forest does NOT learn what
"normal" looks like directly, and it does NOT use the is_anomaly labels
during training at all. Instead, it exploits one simple insight:

    Anomalies are easier to isolate than normal points, because they are
    rare and sit apart from the dense cluster of typical behaviour.

What happens during model.fit(X):

  1. The model builds many random decision trees (n_estimators=200 below).
     Each tree is built independently, using randomness.

  2. For each tree, the algorithm repeatedly:
       - picks a random feature (e.g. hour_sin)
       - picks a random split value within that feature's range
       - divides the data into two groups
       - repeats on each smaller group, until every point is alone
         in its own group ("isolated")

  3. The algorithm records how many splits it took to isolate each point.

       - Anomalies (rare, unusual feature combinations - e.g. accessed
         outside shift hours) tend to get isolated QUICKLY, in few splits,
         because they differ from the bulk of the data.

       - Normal points (the dense cluster of typical shift-hours,
         ward-matched events) take MANY MORE splits to isolate, because
         they are surrounded by lots of similar points.

  4. Averaged across all 200 trees, a SHORT average path length means
     "isolated easily" -> high anomaly score. A LONG average path length
     means "hard to isolate" -> low anomaly score (more normal).

This is why it is called "unsupervised": the is_anomaly column is only
used AFTERWARDS, to check how well the model's own judgement matches
reality. It is never shown to the model during model.fit(X).

Run from the project root:
    py src/anomaly_model.py
================================================================================
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

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from feature_engineering import FEATURE_COLS      # the ONE place the model's feature list is defined


def train_isolation_forest(df, feature_cols=FEATURE_COLS, contamination=0.05, seed=42):
    """
    Trains the Isolation Forest model.

    Parameters
    ----------
    contamination : float
        The assumed proportion of anomalous events in the data (here, 5%,
        matching the anomaly_rate used when the dataset was generated).
        This does NOT change how the random trees are built - it only
        sets where the final normal/anomaly cutoff line is drawn once
        every point has an anomaly score.
    seed : int
        Fixes the randomness so the same trees are built every run,
        supporting reproducibility (see Chapter 3.7).
    """
    X = df[feature_cols].fillna(0)

    model = IsolationForest(
        n_estimators=200,      # number of random trees to build
        contamination=contamination,
        random_state=seed,     # reproducibility
    )
    model.fit(X)                # <-- labels are NEVER passed in here

    # sklearn convention: decision_function is HIGHER for normal points.
    # We flip the sign so higher anomaly_score = more anomalous (more
    # intuitive for reporting and for the orchestration logic later).
    raw_scores = model.decision_function(X)
    predictions = model.predict(X)   # -1 = anomaly, 1 = normal

    df = df.copy()
    df["anomaly_score"] = -raw_scores
    df["predicted_anomaly"] = (predictions == -1).astype(int)

    return model, df


def evaluate_model(df):
    """Compares the model's unsupervised predictions against the known
    ground-truth labels (is_anomaly), which were withheld during training."""
    y_true = df["is_anomaly"]
    y_pred = df["predicted_anomaly"]

    print("=== Confusion Matrix ===")
    print("             Predicted Normal  Predicted Anomaly")
    cm = confusion_matrix(y_true, y_pred)
    print(f"Actual Normal      {cm[0][0]:>10}       {cm[0][1]:>10}")
    print(f"Actual Anomaly     {cm[1][0]:>10}       {cm[1][1]:>10}")

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

    auc = roc_auc_score(y_true, df["anomaly_score"])
    print(f"ROC-AUC Score: {auc:.4f}  (1.0 = perfect separation, 0.5 = random guessing)")

    print("\n=== Anomaly score by true label (sanity check) ===")
    print(df.groupby("is_anomaly")["anomaly_score"].agg(["mean", "std"]).round(4))

    print("\n=== Anomaly score by anomaly TYPE ===")
    print(df.groupby("anomaly_type")["anomaly_score"].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    df = pd.read_csv("data/access_logs_features.csv")

    model, scored_df = train_isolation_forest(df)

    scored_df.to_csv("data/access_logs_scored.csv", index=False)
    print(f"Scored data saved -> data/access_logs_scored.csv\n")

    evaluate_model(scored_df)

    import joblib
    joblib.dump(model, "models/isolation_forest.pkl")
    print("\nModel saved -> models/isolation_forest.pkl")
