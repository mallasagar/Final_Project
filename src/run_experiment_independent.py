import sys
from pathlib import Path


def _add_script_folders():
    """Let this script find the other pipeline scripts wherever they sit (src/ or src/model/)."""
    here = Path(__file__).resolve().parent
    skip = {"venv", "__pycache__", "data", "results", "models"}
    for base in (here, here.parent):
        for p in [base] + [d for d in base.iterdir() if d.is_dir() and d.name not in skip and not d.name.startswith(".")]:
            if str(p) not in sys.path:
                sys.path.append(str(p))


_add_script_folders()

import joblib
import numpy as np
import pandas as pd

from access_log_generator import generate_access_logs, EVENTS_PER_DAY
from feature_engineering import engineer_features, FEATURE_COLS
from orchestration import orchestrate
from run_experiment import simulate_condition, compute_trial_metrics, CONDITIONS, SCENARIOS, DEFAULT_WINDOW

N_TRIALS = 20
TRIAL_DAYS = 30          # each trial is a fresh 30-day log
BASE_SEED = 1000         # trial t uses seed 1000+t; the model was trained on seed 42


def make_trial_dataset(seed, staff_df, patients_df, model):
    logs = generate_access_logs(patients_df, staff_df, n_days=TRIAL_DAYS, events_per_day=EVENTS_PER_DAY, seed=seed)
    df, _ = engineer_features(logs, staff_df)
    X = df[FEATURE_COLS].fillna(0)
    df["anomaly_score"] = -model.decision_function(X)           # the trained model is only applied, never refit
    df["predicted_anomaly"] = (model.predict(X) == -1).astype(int)
    return orchestrate(df, seed=seed)


def dataset_metrics(df, seed):
    """Overall classification metrics for each condition on one whole dataset (both scenarios together)."""
    out = []
    for ci, condition in enumerate(CONDITIONS):
        rng = np.random.default_rng([seed, 99, ci])
        sim = simulate_condition(df, condition, rng, DEFAULT_WINDOW)
        y, f = sim["is_anomaly"].to_numpy(), sim["detected"].to_numpy()
        tp, fp = int(((f == 1) & (y == 1)).sum()), int(((f == 1) & (y == 0)).sum())
        tn, fn = int(((f == 0) & (y == 0)).sum()), int(((f == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn)
        out.append({"seed": seed, "condition": condition,
                    "accuracy_%": 100 * (tp + tn) / len(y),
                    "precision_%": 100 * precision,
                    "recall_%": 100 * recall,
                    "F1_%": 100 * 2 * precision * recall / (precision + recall) if tp else np.nan,
                    "false_positive_%": 100 * fp / (fp + tn),
                    "all_normal_baseline_%": 100 * (1 - y.mean())})
    return out


def run_independent(staff_df, patients_df, model, n_trials=N_TRIALS):
    rows, class_rows = [], []
    for trial in range(n_trials):
        seed = BASE_SEED + trial
        df = make_trial_dataset(seed, staff_df, patients_df, model)
        class_rows.extend(dataset_metrics(df, seed))
        for si, scenario in enumerate(SCENARIOS):
            scenario_df = df[(df["anomaly_type"] == scenario) | (df["anomaly_type"] == "none")]
            for ci, condition in enumerate(CONDITIONS):
                rng = np.random.default_rng([seed, si, ci])
                metrics = compute_trial_metrics(simulate_condition(scenario_df, condition, rng, DEFAULT_WINDOW))
                rows.append({"trial": trial, "condition": condition, "scenario": scenario, "seed": seed, **metrics})
    return pd.DataFrame(rows), pd.DataFrame(class_rows)


if __name__ == "__main__":
    staff_df = pd.read_csv("data/staff.csv")
    patients_df = pd.read_csv("data/patients_with_wards.csv")
    model = joblib.load("models/isolation_forest.pkl")

    results, class_df = run_independent(staff_df, patients_df, model)
    class_df.round(3).to_csv("results/classification_metrics_independent.csv", index=False)
    results.to_csv("results/experiment_results_independent.csv", index=False)
    print(f"{N_TRIALS} independent 30-day datasets x {len(SCENARIOS)} scenarios x {len(CONDITIONS)} conditions "
          f"= {len(results)} rows -> results/experiment_results_independent.csv\n")

    cols = ["MTTD", "MTTR", "detection_rate", "false_positive_rate", "records_exposed", "records_exposed_incl_missed"]
    print("=== Mean metrics by condition (independent datasets) ===")
    print(results.groupby("condition")[cols].mean().round(3).reindex(CONDITIONS).to_string())

    print("\n=== Detection rate (%), mean +/- spread across the 20 independent datasets ===")
    g = results.groupby(["condition", "scenario"])["detection_rate"].agg(["mean", "std"]).round(2)
    print(g.to_string())
    fp = results.groupby("condition")["false_positive_rate"].agg(["mean", "std"]).round(2)
    print("\n=== False-positive rate (%), mean +/- spread ===")
    print(fp.reindex(CONDITIONS).to_string())

    print("\n=== Overall classification metrics (%), mean +/- spread over the 20 independent datasets ===")
    cm = class_df.groupby("condition")[["accuracy_%", "precision_%", "recall_%", "F1_%", "false_positive_%"]].agg(["mean", "std"]).round(1)
    print(cm.reindex(CONDITIONS).to_string())
    print(f"Accuracy of a system that flags nothing: {class_df['all_normal_baseline_%'].mean():.1f}%")
    print("Saved -> results/classification_metrics_independent.csv")

    old = Path("results/experiment_results.csv")
    if old.exists():
        o = pd.read_csv(old).groupby("condition")["detection_rate"].mean().round(2)
        n = results.groupby("condition")["detection_rate"].mean().round(2)
        print("\n=== Same-dataset resampling vs independent datasets: mean detection rate (%) ===")
        print(pd.DataFrame({"resampled (same data)": o, "independent datasets": n}).reindex(CONDITIONS).to_string())
