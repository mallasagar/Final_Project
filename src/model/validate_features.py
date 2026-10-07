"""
validate_features.py
Automated checks that the engineered features are correct before the model is trained.

Run from the project root:
    py src/validate_features.py
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

import numpy as np
import sys
import pandas as pd

from feature_engineering import FEATURE_COLS      # the ONE place the model's feature list is defined


def run_validation(df, raw_df):
    passed, total, issues = 0, 0, []

    def check(name, condition):
        nonlocal passed, total
        total += 1
        passed += bool(condition)
        if not condition:
            issues.append(name)
        print(f"[{'PASS' if condition else 'FAIL'}] {name}")

    print("=" * 66)
    print("FEATURE ENGINEERING VALIDATION")
    print("=" * 66)

    n_missing = int(df[FEATURE_COLS].isna().sum().sum())
    check(f"No missing values in feature columns (found {n_missing})", n_missing == 0)
    check("'ward_match' contains only 0/1 values", set(df["ward_match"].unique()).issubset({0, 1}))
    check("'hours_beyond_shift' is a whole number >= 0",
          (df["hours_beyond_shift"] >= 0).all() and (df["hours_beyond_shift"] % 1 == 0).all())
    check("'hour_sin' and 'hour_cos' are within [-1, 1]",
          df["hour_sin"].between(-1, 1).all() and df["hour_cos"].between(-1, 1).all())
    check("hour_sin^2 + hour_cos^2 = 1 for every row (valid cyclical encoding)",
          np.allclose(df["hour_sin"] ** 2 + df["hour_cos"] ** 2, 1.0, atol=1e-6))
    if "access_count_1h" in df.columns:        # supporting column, not a model feature
        check("'access_count_1h' is always >= 1 (every event counts itself)", (df["access_count_1h"] >= 1).all())

    # recompute hours_beyond_shift independently from the raw hour and shift
    h = pd.to_datetime(df["timestamp"]).dt.hour
    expected = np.where(h >= df["shift_end"], h - df["shift_end"] + 1,
                        np.where(h < df["shift_start"], df["shift_start"] - h, 0))
    check("'hours_beyond_shift' matches an independent recalculation", (df["hours_beyond_shift"] == expected).all())

    normal = df[df["anomaly_type"] == "none"]
    off = df[df["anomaly_type"] == "off_hours"]
    cross = df[df["anomaly_type"] == "cross_ward"]

    pct = (off["hours_beyond_shift"] > 0).mean()
    check(f"off_hours anomalies are outside the shift ({pct * 100:.1f}%)", pct > 0.95)
    pct = (cross["ward_match"] == 0).mean()
    check(f"cross_ward anomalies have ward_match=0 ({pct * 100:.1f}%)", pct > 0.95)
    pct = (cross["hours_beyond_shift"] == 0).mean()
    check(f"cross_ward anomalies happen inside the shift ({pct * 100:.1f}%)", pct > 0.95)
    pct = (normal["ward_match"] == 1).mean()
    check(f"legitimate events mostly have ward_match=1 ({pct * 100:.1f}%)", pct > 0.90)
    pct = (normal["hours_beyond_shift"] <= 1).mean()
    check(f"legitimate events are at most 1 hour outside the shift ({pct * 100:.1f}%)", pct > 0.99)
    check(f"No rows lost during feature engineering ({len(df)} vs {len(raw_df)} raw)", len(df) == len(raw_df))

    print("=" * 66)
    print(f"RESULT: {passed}/{total} checks passed")
    print("All validation checks passed." if not issues else f"Issues found in: {issues}")
    print(f"For reference: legitimate events max {normal['hours_beyond_shift'].max()} hour(s) outside the shift; "
          f"off-hours anomalies min {off['hours_beyond_shift'].min()}.")
    print("=" * 66)
    return passed, total, issues


if __name__ == "__main__":
    issues = run_validation(pd.read_csv("data/access_logs_features.csv"), pd.read_csv("data/access_logs.csv"))[2]
    sys.exit(1 if issues else 0)      # non-zero exit if any check failed
