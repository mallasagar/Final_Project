import numpy as np
import pandas as pd

N_DAYS = 90
EVENTS_PER_DAY = 240            # about 5 events per person per day with 48 staff
ANOMALY_RATE = 0.05
OUTSIDE_SHIFT_RATE = 0.10       # legitimate events one hour before start / after end
LEGIT_CROSS_WARD_RATE = 0.05    # legitimate cover on another ward
OFF_HOURS_MARGIN = 3            # hours kept clear between the shift and the off-hours anomaly window
START_DATE = "2026-01-01"
SEED = 42


def generate_access_logs(patients_df, staff_df, n_days=N_DAYS, events_per_day=EVENTS_PER_DAY,
                         anomaly_rate=ANOMALY_RATE, seed=SEED):
    rng = np.random.default_rng(seed)
    n = n_days * events_per_day
    staff = staff_df.reset_index(drop=True)

    # --- who does each event ---
    idx = rng.integers(0, len(staff), n)
    user_id = staff["user_id"].to_numpy()[idx]
    role = staff["role"].to_numpy()[idx]
    ward = staff["assigned_ward"].to_numpy()[idx]
    start = staff["shift_start"].to_numpy()[idx].astype(int)
    end = staff["shift_end"].to_numpy()[idx].astype(int)

    # --- which events are anomalies ---
    is_anom = rng.random(n) < anomaly_rate
    atype = np.where(is_anom, np.where(rng.random(n) < 0.5, "off_hours", "cross_ward"), "none")
    normal = ~is_anom
    off = atype == "off_hours"
    cross = atype == "cross_ward"

    # --- hour of access ---
    hour = rng.integers(start, end)                                   # inside the shift (end excluded)
    outside = normal & (rng.random(n) < OUTSIDE_SHIFT_RATE)           # legitimate, one hour outside
    hour = np.where(outside, np.where(rng.random(n) < 0.5, start - 1, end), hour)

    before = np.maximum(start - OFF_HOURS_MARGIN, 0)                  # hours 0 .. start-margin-1
    after = np.maximum(24 - (end + OFF_HOURS_MARGIN), 0)              # hours end+margin .. 23
    if np.any(before + after == 0):
        raise ValueError("Shift leaves no room for off-hours anomalies; reduce OFF_HOURS_MARGIN.")
    k = rng.integers(0, np.maximum(before + after, 1))
    off_hour = np.where(k < before, k, end + OFF_HOURS_MARGIN + (k - before))
    hour = np.where(off, off_hour, hour)

    # --- which patient: own ward, or another ward for cross-ward events and legitimate cover ---
    want_other = cross | (normal & (rng.random(n) < LEGIT_CROSS_WARD_RATE))
    patient_id = np.empty(n, dtype=object)
    patient_ward = np.empty(n, dtype=object)
    for w in staff["assigned_ward"].unique():
        own_pool = patients_df[patients_df["ward"] == w]
        other_pool = patients_df[patients_df["ward"] != w]
        if len(own_pool) == 0 or len(other_pool) == 0:
            raise ValueError(f"Ward '{w}' has no patients (or is the only ward). Check the ward lists match.")
        for flag, pool in [(False, own_pool), (True, other_pool)]:
            sel = (ward == w) & (want_other == flag)
            pick = rng.integers(0, len(pool), int(sel.sum()))
            patient_id[sel] = pool["patient_id"].to_numpy()[pick]
            patient_ward[sel] = pool["ward"].to_numpy()[pick]

    # --- timestamp and action ---
    day = np.repeat(np.arange(n_days), events_per_day)
    ts = (pd.Timestamp(START_DATE) + pd.to_timedelta(day, "D")
          + pd.to_timedelta(hour, "h") + pd.to_timedelta(rng.integers(0, 60, n), "m"))
    action = rng.choice(["read", "read", "read", "write", "export"], n)

    df = pd.DataFrame({
        "timestamp": ts, "user_id": user_id, "role": role, "assigned_ward": ward,
        "patient_id": patient_id, "patient_ward": patient_ward, "action": action,
        "is_anomaly": is_anom.astype(int), "anomaly_type": atype,
    })
    return df.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    patients_df = pd.read_csv("data/patients_with_wards.csv")
    staff_df = pd.read_csv("data/staff.csv")

    logs_df = generate_access_logs(patients_df, staff_df)
    logs_df.to_csv("data/access_logs.csv", index=False)

    print(f"Generated {len(logs_df)} total access events -> data/access_logs.csv")
    print(f"Anomalies: {logs_df['is_anomaly'].sum()} ({logs_df['is_anomaly'].mean() * 100:.1f}%)")
    print("\nAnomaly type breakdown:")
    print(logs_df["anomaly_type"].value_counts())
    normal = logs_df[logs_df["is_anomaly"] == 0]
    hours = pd.to_datetime(normal["timestamp"]).dt.hour
    print(f"\nLegitimate events outside the shift (one hour either side): {(~hours.between(9, 16)).sum()}")
    print(f"Legitimate cross-ward cover events: {(normal['assigned_ward'] != normal['patient_ward']).sum()}")
