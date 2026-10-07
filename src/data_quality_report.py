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

import argparse
import sys

import numpy as np
import pandas as pd

from feature_engineering import FEATURE_COLS      # the ONE place the model's feature list is defined
LABEL_COLS = ["is_anomaly", "anomaly_type"]
MAX_EXACT_DUPLICATES = 5          # random coincidence at this data size gives about 0-2
MAX_PLAUSIBLE_HOURLY_COUNT = 15   # one user opening 15+ records in an hour would be suspicious


def audit(df, raw_rows=None):
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    print("=" * 74)
    print(f"DATA QUALITY REPORT  -  {len(df)} rows, {df.shape[1]} columns")
    print("=" * 74)

    # ---- 1. completeness ----
    print("\n-- Completeness --")
    nulls = df.isna().sum()
    check("No missing values in any column", nulls.sum() == 0,
          "none" if nulls.sum() == 0 else f"{nulls[nulls > 0].to_dict()}")
    numeric = df[FEATURE_COLS].select_dtypes(include=[np.number])
    check("All feature columns are numeric", len(numeric.columns) == len(FEATURE_COLS))
    check("No infinite values in the features", not np.isinf(df[FEATURE_COLS]).any().any())

    # ---- 2. duplicates ----
    print("\n-- Duplicates --")
    n_exact = int(df.duplicated().sum())
    check(f"Exact duplicate rows within random coincidence (<= {MAX_EXACT_DUPLICATES})", n_exact <= MAX_EXACT_DUPLICATES,
          f"found {n_exact}")
    ties = int(df.duplicated(subset=["user_id", "timestamp"], keep=False).sum())
    print(f"[INFO] Rows sharing a user and minute with another row: {ties} ({100 * ties / len(df):.2f}%) "
          "- expected at this volume; it only nudges access_count_1h by 1 for those rows")
    unique_vectors = len(df[FEATURE_COLS].drop_duplicates())
    print(f"[INFO] Distinct feature vectors: {unique_vectors} of {len(df)} rows. Many events look alike because "
          "the features take few values; identical rows are normal here")

    # ---- 3. value ranges ----
    print("\n-- Value ranges --")
    check("ward_match is only 0 or 1", set(df["ward_match"].unique()) <= {0, 1})
    check("hours_beyond_shift is a whole number between 0 and 23",
          df["hours_beyond_shift"].between(0, 23).all() and (df["hours_beyond_shift"] % 1 == 0).all())
    check("hour_sin and hour_cos are within [-1, 1]", df["hour_sin"].between(-1, 1).all() and df["hour_cos"].between(-1, 1).all())
    check("hour_sin^2 + hour_cos^2 = 1 for every row", np.allclose(df["hour_sin"] ** 2 + df["hour_cos"] ** 2, 1.0, atol=1e-6))
    if "access_count_1h" in df.columns:        # supporting column, not a model feature
        check("access_count_1h is at least 1 (every event counts itself)", (df["access_count_1h"] >= 1).all())
        check(f"access_count_1h has no implausible outliers (max <= {MAX_PLAUSIBLE_HOURLY_COUNT})",
              df["access_count_1h"].max() <= MAX_PLAUSIBLE_HOURLY_COUNT, f"max {df['access_count_1h'].max():.0f}")
    check("Every feature varies (none is constant)", (df[FEATURE_COLS].nunique() > 1).all())

    # ---- 4. labels and leakage ----
    print("\n-- Labels and leakage --")
    check("is_anomaly is only 0 or 1", set(df["is_anomaly"].unique()) <= {0, 1})
    check("is_anomaly agrees with anomaly_type",
          ((df["is_anomaly"] == 0) == (df["anomaly_type"] == "none")).all())
    check("No label column is used as a model feature", not set(FEATURE_COLS) & set(LABEL_COLS))
    rate = df["is_anomaly"].mean()
    check("Anomaly rate is between 2% and 10%", 0.02 <= rate <= 0.10, f"{100 * rate:.2f}%")
    if raw_rows is not None:
        check("No rows lost since the raw access log", len(df) == raw_rows, f"{len(df)} vs {raw_rows}")

    # ---- 5. noise ----
    print("\n-- Noise (injected on purpose, and checked for anything unintended) --")
    normal = df[df["anomaly_type"] == "none"]
    off = df[df["anomaly_type"] == "off_hours"]
    outside = (normal["hours_beyond_shift"] >= 1).mean()
    cover = (normal["ward_match"] == 0).mean()
    print(f"[INFO] Legitimate events one hour outside the shift: {100 * outside:.1f}%   (designed: 10%)")
    print(f"[INFO] Legitimate events on another ward (cover):    {100 * cover:.1f}%   (designed: 5%)")
    check("Injected noise is close to its design values",
          abs(outside - 0.10) < 0.02 and abs(cover - 0.05) < 0.02)
    check("No UNINTENDED noise: legitimate events are never more than 1 hour outside the shift",
          normal["hours_beyond_shift"].max() <= 1, f"max {normal['hours_beyond_shift'].max():.0f}")
    check("Off-hours anomalies are clearly separate from legitimate lateness",
          off["hours_beyond_shift"].min() > normal["hours_beyond_shift"].max(),
          f"anomalies from {off['hours_beyond_shift'].min():.0f} hours, legitimate up to {normal['hours_beyond_shift'].max():.0f}")

    print("\n-- Feature summary --")
    print(df[FEATURE_COLS].describe().loc[["min", "mean", "max"]].round(3).to_string())
    failed = results.count(False)
    print("\n" + "=" * 74)
    print(f"RESULT: {len(results) - failed}/{len(results)} checks passed" + ("" if not failed else f"  -  {failed} FAILED"))
    print("=" * 74)
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/access_logs_features.csv")
    parser.add_argument("--raw", default="data/access_logs.csv")
    args = parser.parse_args()
    try:
        data = pd.read_csv(args.file)
    except FileNotFoundError:
        sys.exit(f"File not found: {args.file}\nRun feature_engineering.py first, from the project root.")
    try:
        raw_rows = len(pd.read_csv(args.raw))
    except FileNotFoundError:
        raw_rows = None
    sys.exit(1 if audit(data, raw_rows) else 0)
