"""
contamination_sweep.py
Shows how the Isolation Forest 'contamination' setting trades recall
against false positives on YOUR data, and what it costs to reach a
target recall (default 90%).

Run from the project root, after feature_engineering.py:
    py src/contamination_sweep.py
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
import pandas as pd, numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, roc_auc_score

from feature_engineering import FEATURE_COLS      # the ONE place the model's feature list is defined
FEATURES = FEATURE_COLS
TARGET_RECALL = 0.90

df = pd.read_csv("data/access_logs_features.csv")
X, y = df[FEATURES].fillna(0), df["is_anomaly"].values
print(f"{len(df)} events, {y.sum()} anomalies. Accuracy of predicting 'all normal': {100*(1-y.mean()):.1f}%\n")

print(f"{'contamination':<14}{'accuracy':<10}{'recall':<9}{'precision':<11}{'FP rate':<9}{'flagged':<8}")
for c in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    pred = (IsolationForest(n_estimators=200, contamination=c, random_state=42).fit(X).predict(X) == -1).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    print(f"{c:<14}{100*(tp+tn)/len(y):<10.1f}{100*tp/(tp+fn):<9.1f}{100*tp/max(tp+fp,1):<11.1f}{100*fp/(fp+tn):<9.1f}{pred.sum():<8d}")

score = -IsolationForest(n_estimators=200, contamination=0.05, random_state=42).fit(X).decision_function(X)
fpr, tpr, _ = roc_curve(y, score)
i = int(np.argmax(tpr >= TARGET_RECALL))
flagged = fpr[i] * (y == 0).sum() + tpr[i] * y.sum()
print(f"\nROC-AUC: {roc_auc_score(y, score):.4f}")
print(f"To reach {TARGET_RECALL:.0%} recall: false-positive rate = {fpr[i]:.1%}, "
      f"flagged = {flagged/len(y):.1%} of events, precision = {tpr[i]*y.sum()/flagged:.1%}")
