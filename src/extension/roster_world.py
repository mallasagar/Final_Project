"""
roster_world.py   (EXTENSION - separate from the main experiment)
A simulated hospital in which legitimate cross-ward access comes from a ROSTER.

Order of construction (this is what keeps the roster from being a shortcut):
  1. The roster is created first (roster_generator.py): who is rostered to which ward on which day.
  2. Legitimate cross-ward access is then generated from it:
       - ROSTERED COVER: on a day a nurse or doctor is rostered to another ward, 90% of their events are on that ward.
       - AD-HOC access: a small share of events by doctors and nurses are on a ward that is neither their home ward nor
         their rostered ward (consults, emergencies). These are legitimate but NOT on the roster.
  3. Anomalies are drawn independently of the roster:
       - off-hours: as in the main experiment (hours 00-05 and 20-23, own ward);
       - cross-ward: inside the shift, on a ward that is neither the user's home ward nor their rostered ward.
  4. The model never sees a label. Its only roster information is a lookup:
       authorised_ward = 1 if the patient is on the user's home ward OR on the ward they are rostered to that day.
Everything else (240 events a day, 5% anomalies, 10% legitimate events one hour outside the shift) is as in the main design.
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

from roster_generator import generate_roster, START_DATE
from feature_engineering import engineer_features

EVENTS_PER_DAY = 240
ANOMALY_RATE = 0.05
OUTSIDE_SHIFT_RATE = 0.10
OFF_HOURS_MARGIN = 3
COVER_DAY_ON_ROSTERED_WARD = 0.90        # on a rostered cover day, share of a person's events on the rostered ward
ADHOC_RATE = 0.015                       # ASSUMPTION: share of clinical staff events that are legitimate but NOT rostered
ROSTER_FEATURES = ["hours_beyond_shift", "authorised_ward", "hour_sin", "hour_cos"]
NO_ROSTER_FEATURES = ["hours_beyond_shift", "ward_match", "hour_sin", "hour_cos"]


def generate_world(patients_df, staff_df, n_days, seed, adhoc_rate=ADHOC_RATE, events_per_day=EVENTS_PER_DAY, anomaly_rate=ANOMALY_RATE):
    rng = np.random.default_rng(seed)
    staff = staff_df.reset_index(drop=True); n_staff = len(staff)
    roster = generate_roster(staff, n_days=n_days, seed=seed)                                   # STEP 1: the roster comes first
    grid = roster.pivot(index="date", columns="user_id", values="rostered_ward").reindex(columns=staff["user_id"])
    R = grid.to_numpy()                                                                          # (day, staff) -> ward rostered that day
    n = n_days * events_per_day
    day = np.repeat(np.arange(n_days), events_per_day)
    idx = rng.integers(0, n_staff, n)
    user_id, role, home = staff["user_id"].to_numpy()[idx], staff["role"].to_numpy()[idx], staff["assigned_ward"].to_numpy()[idx]
    start, end = staff["shift_start"].to_numpy()[idx].astype(int), staff["shift_end"].to_numpy()[idx].astype(int)
    rost = R[day, idx]

    # anomalies are drawn independently of the roster
    is_anom = rng.random(n) < anomaly_rate
    atype = np.where(is_anom, np.where(rng.random(n) < 0.5, "off_hours", "cross_ward"), "none")
    normal, off, cross = ~is_anom, atype == "off_hours", atype == "cross_ward"

    hour = rng.integers(start, end)
    outside = normal & (rng.random(n) < OUTSIDE_SHIFT_RATE)
    hour = np.where(outside, np.where(rng.random(n) < 0.5, start - 1, end), hour)
    before, after = np.maximum(start - OFF_HOURS_MARGIN, 0), np.maximum(24 - (end + OFF_HOURS_MARGIN), 0)
    k = rng.integers(0, np.maximum(before + after, 1))
    hour = np.where(off, np.where(k < before, k, end + OFF_HOURS_MARGIN + (k - before)), hour)

    # which ward's patient is opened                                                             # STEP 2 and 3
    target = home.copy()
    on_cover = normal & (rost != home) & (rng.random(n) < COVER_DAY_ON_ROSTERED_WARD)             # rostered cover (legitimate)
    target[on_cover] = rost[on_cover]
    adhoc = normal & ~on_cover & (role != "Admin") & (rng.random(n) < adhoc_rate)                  # ad-hoc, not on the roster (legitimate)
    wards = sorted(staff["assigned_ward"].unique())
    for i in np.where(adhoc | cross)[0]:                                                          # ad-hoc access and cross-ward anomalies: an UNROSTERED ward
        choices = [w for w in wards if w != home[i] and w != rost[i]]
        target[i] = choices[rng.integers(0, len(choices))]
    patient_id, patient_ward = np.empty(n, dtype=object), np.empty(n, dtype=object)
    for w in wards:
        pool = patients_df[patients_df["ward"] == w]; sel = target == w
        pick = rng.integers(0, len(pool), int(sel.sum()))
        patient_id[sel], patient_ward[sel] = pool["patient_id"].to_numpy()[pick], pool["ward"].to_numpy()[pick]

    ts = (pd.Timestamp(START_DATE) + pd.to_timedelta(day, "D") + pd.to_timedelta(hour, "h") + pd.to_timedelta(rng.integers(0, 60, n), "m"))
    action = rng.choice(["read", "read", "read", "write", "export"], n)
    logs = pd.DataFrame({"timestamp": ts, "user_id": user_id, "role": role, "assigned_ward": home, "patient_id": patient_id, "patient_ward": patient_ward,
                         "action": action, "is_anomaly": is_anom.astype(int), "anomaly_type": atype}).sort_values("timestamp").reset_index(drop=True)
    return logs, roster


def engineer_roster_features(logs_df, staff_df, roster_df):
    """The main features, plus authorised_ward from a plain roster LOOKUP (no label is used)."""
    df, _ = engineer_features(logs_df, staff_df)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
    df = df.merge(roster_df[["date", "user_id", "rostered_ward"]], on=["date", "user_id"], how="left")
    if df["rostered_ward"].isna().any():
        raise ValueError("Some events have no roster entry for that user and day.")
    df["authorised_ward"] = ((df["patient_ward"] == df["assigned_ward"]) | (df["patient_ward"] == df["rostered_ward"])).astype(int)
    return df
