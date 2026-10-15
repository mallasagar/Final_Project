"""
access_log_generator.py
Generates the labelled access log: who opened whose record, and when.

The hospital has a ROSTER (data/roster.csv, generated with the staff by staff_generator.py): every staff member works
09:00-17:00 every day, on their home ward unless they are rostered to cover a compatible ward (see roster_generator.py).
Legitimate cross-ward access therefore comes from the roster.

Design
  Normal events      Hour inside the shift. 10% are legitimate "slightly outside" events, one hour before the shift starts
                     or one hour after it ends.
  Rostered cover     On a day a nurse or doctor is rostered to another ward, 90% of their events are on that ward (legitimate).
  Ad-hoc access      ADHOC_RATE of the events of doctors and nurses are on a ward that is neither their home ward nor their
                     rostered ward (consults, emergencies). Legitimate, but NOT on the roster. (An assumption.)
  Off-hours anomaly  An hour more than OFF_HOURS_MARGIN hours outside the shift (hours 0-5 and 20-23), own-ward patient.
  Cross-ward anomaly An hour inside the shift, on a ward that is neither the user's home ward nor their rostered ward.
Anomalies are drawn independently of the roster. Each event carries 'rostered_ward': the ward the roster assigns that user
that day. It is a plain lookup (what the roster says), never a label.

Run from the project root:
    py src/access_log_generator.py
"""
import sys
from pathlib import Path


def _add_script_folders():
    here = Path(__file__).resolve().parent
    skip = {"venv", "__pycache__", "data", "results", "models", "logs"}
    for base in (here, here.parent):
        for p in [base] + [d for d in base.iterdir() if d.is_dir() and d.name not in skip and not d.name.startswith(".")]:
            if str(p) not in sys.path:
                sys.path.append(str(p))


_add_script_folders()

import numpy as np
import pandas as pd

from roster_generator import generate_roster

N_DAYS = 90
EVENTS_PER_DAY = 240            # about 5 events per person per day with 48 staff
ANOMALY_RATE = 0.05
OUTSIDE_SHIFT_RATE = 0.10       # legitimate events one hour before start / after end
ADHOC_RATE = 0.015              # legitimate access by doctors and nurses on a ward that is NOT on their roster (assumption)
COVER_DAY_ON_ROSTERED_WARD = 0.90   # on a rostered cover day, share of a person's events on the rostered ward
OFF_HOURS_MARGIN = 3            # hours kept clear between the shift and the off-hours anomaly window
START_DATE = "2026-01-01"
SEED = 42


def generate_access_logs(patients_df, staff_df, n_days=N_DAYS, events_per_day=EVENTS_PER_DAY,
                         anomaly_rate=ANOMALY_RATE, seed=SEED, roster_df=None):
    rng = np.random.default_rng(seed)
    staff = staff_df.reset_index(drop=True); n_staff = len(staff)
    roster = roster_df if roster_df is not None else generate_roster(staff, n_days=n_days, seed=seed, start_date=START_DATE)
    grid = roster.pivot(index="date", columns="user_id", values="rostered_ward").reindex(columns=staff["user_id"])
    R = grid.to_numpy()                                              # (day, staff) -> ward rostered that day
    n = n_days * events_per_day
    day = np.repeat(np.arange(n_days), events_per_day)

    # --- who does each event ---
    idx = rng.integers(0, n_staff, n)
    user_id, role, home = staff["user_id"].to_numpy()[idx], staff["role"].to_numpy()[idx], staff["assigned_ward"].to_numpy()[idx]
    start, end = staff["shift_start"].to_numpy()[idx].astype(int), staff["shift_end"].to_numpy()[idx].astype(int)
    rost = R[day, idx]

    # --- which events are anomalies (independent of the roster) ---
    is_anom = rng.random(n) < anomaly_rate
    atype = np.where(is_anom, np.where(rng.random(n) < 0.5, "off_hours", "cross_ward"), "none")
    normal, off, cross = ~is_anom, atype == "off_hours", atype == "cross_ward"

    # --- hour of access ---
    hour = rng.integers(start, end)                                   # inside the shift (end excluded)
    outside = normal & (rng.random(n) < OUTSIDE_SHIFT_RATE)           # legitimate, one hour outside
    hour = np.where(outside, np.where(rng.random(n) < 0.5, start - 1, end), hour)
    before = np.maximum(start - OFF_HOURS_MARGIN, 0)                  # hours 0 .. start-margin-1
    after = np.maximum(24 - (end + OFF_HOURS_MARGIN), 0)              # hours end+margin .. 23
    if np.any(before + after == 0):
        raise ValueError("Shift leaves no room for off-hours anomalies; reduce OFF_HOURS_MARGIN.")
    k = rng.integers(0, np.maximum(before + after, 1))
    hour = np.where(off, np.where(k < before, k, end + OFF_HOURS_MARGIN + (k - before)), hour)

    # --- which ward's patient is opened ---
    target = home.copy()
    on_cover = normal & (rost != home) & (rng.random(n) < COVER_DAY_ON_ROSTERED_WARD)   # rostered cover
    target[on_cover] = rost[on_cover]
    adhoc = normal & ~on_cover & (role != "Admin") & (rng.random(n) < ADHOC_RATE)         # ad-hoc, not rostered
    wards = sorted(staff["assigned_ward"].unique())
    for i in np.where(adhoc | cross)[0]:                              # ad-hoc access and cross-ward anomalies: an UNROSTERED ward
        choices = [w for w in wards if w != home[i] and w != rost[i]]
        target[i] = choices[rng.integers(0, len(choices))]
    patient_id, patient_ward = np.empty(n, dtype=object), np.empty(n, dtype=object)
    for w in wards:
        pool = patients_df[patients_df["ward"] == w]
        if len(pool) == 0:
            raise ValueError(f"Ward '{w}' has no patients. Check the ward lists match.")
        sel = target == w
        pick = rng.integers(0, len(pool), int(sel.sum()))
        patient_id[sel], patient_ward[sel] = pool["patient_id"].to_numpy()[pick], pool["ward"].to_numpy()[pick]

    # --- timestamp and action ---
    ts = (pd.Timestamp(START_DATE) + pd.to_timedelta(day, "D") + pd.to_timedelta(hour, "h") + pd.to_timedelta(rng.integers(0, 60, n), "m"))
    action = rng.choice(["read", "read", "read", "write", "export"], n)

    df = pd.DataFrame({
        "timestamp": ts, "user_id": user_id, "role": role, "assigned_ward": home, "rostered_ward": rost,
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
    mism = normal[normal["assigned_ward"] != normal["patient_ward"]]
    rostered = (mism["patient_ward"] == mism["rostered_ward"])
    print(f"\nLegitimate events outside the shift (one hour either side): {(~hours.between(9, 16)).sum()}")
    print(f"Legitimate cross-ward events: {len(mism)}  (on the roster: {int(rostered.sum())}, ad-hoc and not rostered: {int((~rostered).sum())})")
