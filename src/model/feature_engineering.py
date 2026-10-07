"""
feature_engineering.py
Turns each access event into numbers for the Isolation Forest.

Features
  hours_beyond_shift  How far outside the user's OWN shift the access happened, in hours:
                      0 inside the shift, 1 for the first hour before the start or after the end,
                      2 for the second, and so on. shift_end is exclusive: a 09:00-17:00 shift
                      covers 09:00-16:59, so an access at 17:xx is 1 hour beyond the shift.
  ward_match          1 if the patient is on the user's own ward, else 0.
  hour_sin, hour_cos  Time of day as a circle, so 23:00 and 00:00 are close together.
  (access_count_1h    Accesses by this user in the last hour. Calculated but NOT used by the model.)

Shifts must not cross midnight (day shifts only).

Run from the project root:
    py src/feature_engineering.py
"""

import numpy as np
import pandas as pd

# The model's features. This is the ONE place the list is defined; every other script imports it.
# access_count_1h is still calculated (as a column for checks and reference) but is NOT a model feature:
# in this simulation it carries no signal about anomalies, and testing showed removing it improves detection.
FEATURE_COLS = ["hours_beyond_shift", "ward_match", "hour_sin", "hour_cos"]


def engineer_features(logs_df, staff_df):
    if not (staff_df["shift_end"] > staff_df["shift_start"]).all():
        raise ValueError("Every shift_end must be later than shift_start (shifts cannot cross midnight).")

    df = logs_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.merge(staff_df[["user_id", "shift_start", "shift_end"]], on="user_id", how="left")
    if df["shift_start"].isna().any():
        raise ValueError("Some log rows have a user_id that is not in the staff file.")
    df["hour"] = df["timestamp"].dt.hour
    h = df["hour"]

    df["hours_beyond_shift"] = np.where(
        h >= df["shift_end"], h - df["shift_end"] + 1,                 # 17:xx -> 1, 18:xx -> 2, ...
        np.where(h < df["shift_start"], df["shift_start"] - h, 0),     # 08:xx -> 1, 07:xx -> 2, ...
    )
    df["ward_match"] = (df["assigned_ward"] == df["patient_ward"]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["_event"] = 1
    df["access_count_1h"] = (
        df.groupby("user_id", group_keys=False)
        .apply(lambda g: g.set_index("timestamp")["_event"].rolling("1h").count())
        .reset_index(drop=True)
    )
    df = df.drop(columns=["_event"])
    return df, FEATURE_COLS


if __name__ == "__main__":
    logs_df = pd.read_csv("data/access_logs.csv")
    staff_df = pd.read_csv("data/staff.csv")
    featured_df, feature_cols = engineer_features(logs_df, staff_df)
    featured_df.to_csv("data/access_logs_features.csv", index=False)
    print("Feature engineering complete -> data/access_logs_features.csv")
    print(f"Feature columns: {feature_cols}")
    print("\nMean feature values by anomaly type (sanity check):")
    print(featured_df.groupby("anomaly_type")[feature_cols].mean().round(3).to_string())
    normal, off = featured_df[featured_df.anomaly_type == "none"], featured_df[featured_df.anomaly_type == "off_hours"]
    print(f"\nhours_beyond_shift: legitimate events max {normal.hours_beyond_shift.max()}, "
          f"off-hours anomalies min {off.hours_beyond_shift.min()}")
