import argparse

import numpy as np
import pandas as pd

TRADITIONAL_CATCH_PROBABILITY = 0.30
TRADITIONAL_MTTD_RANGE_SEC = (5 * 86400, 9 * 86400)
TRADITIONAL_MTTR_RANGE_SEC = (3600, 5 * 3600)

BASIC_MTTD_RANGE_SEC = (3600, 8 * 3600)
BASIC_MTTR_RANGE_SEC = (1800, 2 * 3600)

# Generic organisational window used by Basic ZTA and Traditional (not each user's own shift).
# Hours START..END inclusive count as normal: (7, 20) means 07:00-20:59.
DEFAULT_WINDOW = (7, 20)
SENSITIVITY_WINDOWS = [(7, 20), (8, 17), (9, 16)]

RECORDS_PER_SECOND_EXPOSED = 1 / 3600     # assumed rate at which an uncontained session reads records

CONDITIONS = ["Traditional", "Basic ZTA", "Enhanced ZTA"]
SCENARIOS = ["off_hours", "cross_ward"]
REQUIRED_COLUMNS = ["hour", "is_anomaly", "anomaly_type", "predicted_anomaly", "response_time_sec"]


def generic_hours_flag(df, window=DEFAULT_WINDOW):
    """Flags access outside the generic window. Looks at the hour only, never the ward."""
    start, end = window
    return ((df["hour"] < start) | (df["hour"] > end)).astype(int)


def simulate_condition(df, condition, rng, window=DEFAULT_WINDOW):
    n = len(df)

    if condition == "Traditional":
        caught = rng.random(n) < TRADITIONAL_CATCH_PROBABILITY
        detected = (generic_hours_flag(df, window) == 1) & caught
        mttd = rng.uniform(*TRADITIONAL_MTTD_RANGE_SEC, size=n)
        mttr = rng.uniform(*TRADITIONAL_MTTR_RANGE_SEC, size=n)
    elif condition == "Basic ZTA":
        detected = generic_hours_flag(df, window) == 1
        mttd = rng.uniform(*BASIC_MTTD_RANGE_SEC, size=n)
        mttr = rng.uniform(*BASIC_MTTR_RANGE_SEC, size=n)
    elif condition == "Enhanced ZTA":
        detected = df["predicted_anomaly"] == 1
        mttd = rng.uniform(0.05, 2.0, size=n)
        mttr = df["response_time_sec"].values
    else:
        raise ValueError(f"Unknown condition: {condition}")

    exposed = np.ceil((mttd + mttr) * RECORDS_PER_SECOND_EXPOSED).clip(min=1)

    # An anomaly this condition never flags is assumed to surface at the next manual audit.
    audit_delay = (rng.uniform(*TRADITIONAL_MTTD_RANGE_SEC, size=n)
                   + rng.uniform(*TRADITIONAL_MTTR_RANGE_SEC, size=n))
    fallback = np.ceil(audit_delay * RECORDS_PER_SECOND_EXPOSED).clip(min=1)
    detected = np.asarray(detected)

    return pd.DataFrame({
        "condition": condition,
        "is_anomaly": df["is_anomaly"].values,
        "anomaly_type": df["anomaly_type"].values,
        "detected": detected.astype(int),
        "mttd_sec": mttd,
        "mttr_sec": mttr,
        "records_exposed": exposed,
        "records_exposed_incl_missed": np.where(detected, exposed, fallback),
    })


def compute_trial_metrics(sim_df):
    anomalies = sim_df[sim_df["is_anomaly"] == 1]
    normal = sim_df[sim_df["is_anomaly"] == 0]
    caught = anomalies[anomalies["detected"] == 1]
    has = len(caught) > 0

    return {
        "MTTD": caught["mttd_sec"].mean() if has else np.nan,
        "MTTR": caught["mttr_sec"].mean() if has else np.nan,
        "detection_rate": 100 * len(caught) / len(anomalies) if len(anomalies) else 0.0,
        "false_positive_rate": 100 * (normal["detected"] == 1).sum() / len(normal) if len(normal) else 0.0,
        "records_exposed": caught["records_exposed"].mean() if has else np.nan,
        "records_exposed_incl_missed": anomalies["records_exposed_incl_missed"].mean() if len(anomalies) else np.nan,
    }


def run_experiment(df, n_trials=20, seed=42, window=DEFAULT_WINDOW):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Run feature_engineering, anomaly_model and orchestration first.")

    results = []
    base_rng = np.random.default_rng(seed)
    for scenario in SCENARIOS:
        scenario_df = df[(df["anomaly_type"] == scenario) | (df["anomaly_type"] == "none")]
        for condition in CONDITIONS:
            for trial in range(n_trials):
                trial_seed = base_rng.integers(0, 1_000_000)
                trial_rng = np.random.default_rng(trial_seed)
                sample = scenario_df.sample(frac=0.5, replace=True, random_state=trial_seed)
                metrics = compute_trial_metrics(simulate_condition(sample, condition, trial_rng, window))
                results.append({"trial": trial, "condition": condition, "scenario": scenario, **metrics})
    return pd.DataFrame(results)


def window_sensitivity(df):
    """How much do Basic ZTA's and Traditional's results depend on the window they are given?"""
    rows = []
    for window in SENSITIVITY_WINDOWS:
        res = run_experiment(df, window=window)
        by_sc = res.groupby(["condition", "scenario"])["detection_rate"].mean()
        by_c = res.groupby("condition")
        rows.append({
            "window": f"{window[0]:02d}:00-{window[1]:02d}:59",
            "basic_off_hours_%": by_sc["Basic ZTA", "off_hours"],
            "basic_cross_ward_%": by_sc["Basic ZTA", "cross_ward"],
            "basic_overall_%": by_c["detection_rate"].mean()["Basic ZTA"],
            "basic_false_pos_%": by_c["false_positive_rate"].mean()["Basic ZTA"],
            "traditional_overall_%": by_c["detection_rate"].mean()["Traditional"],
        })
    return pd.DataFrame(rows).round(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the three-condition experiment.")
    parser.add_argument("--sensitivity", action="store_true", help="also report how the generic window matters")
    args = parser.parse_args()

    df = pd.read_csv("data/access_logs_orchestrated.csv")

    results_df = run_experiment(df, n_trials=20)
    results_df.to_csv("results/experiment_results.csv", index=False)
    print(f"Experiment complete: {len(results_df)} trial results -> results/experiment_results.csv\n")

    print("=== Mean metrics by condition (across both scenarios) ===")
    cols = ["MTTD", "MTTR", "detection_rate", "false_positive_rate", "records_exposed", "records_exposed_incl_missed"]
    print(results_df.groupby("condition")[cols].mean().round(3).reindex(CONDITIONS).to_string())

    print("\n=== Detection rate BY SCENARIO (%) ===")
    print(results_df.groupby(["condition", "scenario"])["detection_rate"].mean().round(2).to_string())

    print("\nNote: MTTD, MTTR and records_exposed are averaged over detected anomalies only, so a condition that")
    print("detects nothing in a scenario (e.g. Basic ZTA on cross-ward) has no value there.")

    if args.sensitivity:
        table = window_sensitivity(df)
        table.to_csv("results/window_sensitivity.csv", index=False)
        print("\n=== Sensitivity: Basic ZTA / Traditional generic window ===")
        print(table.to_string(index=False))
        print("Saved -> results/window_sensitivity.csv")
