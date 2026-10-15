"""
test_datasets.py
STEP 2 of 3 - TEST the trained model on every dataset folder in testing/datasets/.
The model is never retrained. For each dataset it writes, inside that dataset's folder:
    access_logs_scored.csv   features, anomaly score, risk tier, automated response
    results.csv              the three security conditions' metrics on THIS dataset

    py src/testing/test_datasets.py             # test datasets that have no results yet
    py src/testing/test_datasets.py --force     # redo all of them
"""
import argparse
import json

import joblib
import pandas as pd

import _common as C

C.add_script_folders()
from feature_engineering import engineer_features, FEATURE_COLS
from orchestration import orchestrate
from evaluate_new_datasets import dataset_rows


def main():
    ap = argparse.ArgumentParser(description="Test the trained model on each dataset folder.")
    ap.add_argument("--force", action="store_true", help="re-test datasets that already have results")
    a = ap.parse_args()

    folders = sorted(d for d in C.DATASETS.iterdir() if d.is_dir()) if C.DATASETS.exists() else []
    if not folders:
        raise SystemExit(f"No datasets in {C.DATASETS}. Run create_datasets.py first.")
    staff = pd.read_csv(C.ROOT / "data" / "staff.csv")
    model = joblib.load(C.ROOT / "models" / "isolation_forest.pkl")

    print(f"{'dataset':<42}{'Traditional':>12}{'Basic ZTA':>11}{'Enhanced':>10}{'Enh. false pos':>16}    (overall detection %)")
    for f in folders:
        if (f / "results.csv").exists() and not a.force:
            print(f"{f.name:<42}  already tested (use --force to redo)"); continue
        cfg = json.load(open(f / "config.json"))
        df, _ = engineer_features(pd.read_csv(f / "access_logs.csv"), staff)
        X = df[FEATURE_COLS].fillna(0)
        df["anomaly_score"] = -model.decision_function(X)                     # applied only, never refit
        df["predicted_anomaly"] = (model.predict(X) == -1).astype(int)
        df = orchestrate(df, seed=cfg["seed"])
        df.to_csv(f / "access_logs_scored.csv", index=False)
        res = pd.DataFrame(dataset_rows(df, cfg["seed"]))
        res.insert(0, "dataset", f.name); res.insert(1, "profile", cfg["profile"])
        res.insert(2, "events", len(df)); res.insert(3, "anomalies", int(df.is_anomaly.sum()))
        res.round({c: 4 if c == "roc_auc" else 3 for c in res.select_dtypes("number").columns}).to_csv(f / "results.csv", index=False)
        g = res.set_index("condition")
        print(f"{f.name:<42}{g.loc['Traditional', 'detection_overall']:>12.1f}{g.loc['Basic ZTA', 'detection_overall']:>11.1f}"
              f"{g.loc['Enhanced ZTA', 'detection_overall']:>10.1f}{g.loc['Enhanced ZTA', 'false_positive']:>16.2f}")
    print("\nNext: py src/testing/compare_datasets.py")


if __name__ == "__main__":
    main()
