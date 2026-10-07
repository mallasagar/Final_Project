"""
orchestration.py
The response stage of the pipeline. It takes the anomaly score the Isolation Forest gave each event,
sorts the event into a risk tier, picks a proportionate automated action, and records how long that
action takes. That recorded time is the Mean Time to Respond (MTTR) for the Enhanced ZTA condition.

  low risk     -> log_only          (record it, no interruption)
  medium risk  -> step_up_auth      (ask the user to verify again)
  high risk    -> revoke_and_alert  (end the session and alert security)

All actions are simulated in software; the latencies are modelled computational times, not
measurements from live network infrastructure.

Run from the project root:
    py src/orchestration.py
"""

import numpy as np
import pandas as pd

# Thresholds on anomaly_score (higher = more anomalous).
#   HIGH_RISK_THRESHOLD = 0.0 is the model's own decision boundary: strictly above it the Isolation Forest
#       itself calls the event an anomaly, so the high tier is exactly the set of events the model flags. That boundary is set by the model's `contamination` setting,
#       so if you change contamination, recheck this value (calibrate_thresholds() below helps).
#   LOW_RISK_THRESHOLD = -0.05 is an empirical choice: events scoring below it sit well inside the
#       normal cluster. Events between the two thresholds are "medium".
LOW_RISK_THRESHOLD = -0.05
HIGH_RISK_THRESHOLD = 0.0

# Modelled latency in seconds for each action. This generates the MTTR metric.
RESPONSE_LATENCY_SECONDS = {
    "log_only": 0.0,
    "step_up_auth": 1.5,
    "revoke_and_alert": 3.0,
}

ACTION_FOR_TIER = {"low": "log_only", "medium": "step_up_auth", "high": "revoke_and_alert"}


def classify_risk(anomaly_score):
    """Risk tier for a single anomaly score."""
    if anomaly_score < LOW_RISK_THRESHOLD:
        return "low"
    elif anomaly_score <= HIGH_RISK_THRESHOLD:
        return "medium"
    return "high"


def determine_action(risk_tier):
    """Orchestration action for a risk tier."""
    return ACTION_FOR_TIER[risk_tier]


def calibrate_thresholds(scored_df, low_percentile=75, high_percentile=95):
    """Optional helper: suggests thresholds from percentiles of the actual score distribution.
    With contamination=0.05 the 95th percentile is close to the model's own cutoff."""
    low = scored_df["anomaly_score"].quantile(low_percentile / 100)
    high = scored_df["anomaly_score"].quantile(high_percentile / 100)
    print(f"Suggested thresholds from current data: LOW={low:.4f}, HIGH={high:.4f}")
    return low, high


def orchestrate(df, seed=42):
    """Adds risk_tier, action_taken and response_time_sec to every scored event."""
    if "anomaly_score" not in df.columns:
        raise ValueError("No 'anomaly_score' column. Run anomaly_model.py first.")

    rng = np.random.default_rng(seed)
    df = df.copy()

    score = df["anomaly_score"].to_numpy()
    df["risk_tier"] = np.select([score < LOW_RISK_THRESHOLD, score <= HIGH_RISK_THRESHOLD],
                                ["low", "medium"], default="high")
    df["action_taken"] = df["risk_tier"].map(ACTION_FOR_TIER)

    # Small random jitter so repeated actions are not all exactly the same length.
    base_latency = df["action_taken"].map(RESPONSE_LATENCY_SECONDS)
    jitter = rng.normal(loc=0, scale=0.15, size=len(df)).clip(min=-0.3, max=0.3)
    df["response_time_sec"] = (base_latency + jitter).clip(lower=0).round(3)
    return df


def summarize_orchestration(df):
    print("=== Risk Tier Distribution ===")
    print(df["risk_tier"].value_counts())

    print("\n=== Action Taken Distribution ===")
    print(df["action_taken"].value_counts())

    print("\n=== Mean Response Time by Action ===")
    print(df.groupby("action_taken")["response_time_sec"].agg(["mean", "std"]).round(3))

    print("\n=== Orchestration Outcome vs. True Anomaly Label ===")
    true_anomalies = df[df["is_anomaly"] == 1]
    escalated = (true_anomalies["action_taken"] == "revoke_and_alert").mean()
    print(f"True anomalies that triggered revoke_and_alert: {escalated * 100:.1f}%")
    false_escalations = df[(df["is_anomaly"] == 0) & (df["action_taken"] == "revoke_and_alert")]
    print(f"Normal events incorrectly escalated to revoke_and_alert: {len(false_escalations)} "
          f"({len(false_escalations) / len(df) * 100:.2f}%)")

    print("\n=== How many events in each tier are real anomalies ===")
    by_tier = df.groupby("risk_tier")["is_anomaly"].agg(events="size", anomalies="sum")
    by_tier["share_anomalous_%"] = (100 * by_tier["anomalies"] / by_tier["events"]).round(1)
    print(by_tier.reindex(["low", "medium", "high"]).to_string())


if __name__ == "__main__":
    df = pd.read_csv("data/access_logs_scored.csv")
    orchestrated_df = orchestrate(df)
    orchestrated_df.to_csv("data/access_logs_orchestrated.csv", index=False)
    print("Orchestration complete -> data/access_logs_orchestrated.csv\n")
    summarize_orchestration(orchestrated_df)
