"""
check_duplicates.py
Checks the access log for duplicate events and gives a PASS / WARN / FAIL verdict.

Two kinds of duplicate are checked:
  1. Exact duplicates: every column identical. At this data size a few are normal random
     coincidence (same user, minute, patient and action), so a small number is only reported.
  2. Conflicting duplicates: the same user, timestamp, patient and action but DIFFERENT
     anomaly labels. That would put contradictory ground truth into the evaluation, so it FAILS.

Usage (run from the project root):
    py src/check_duplicates.py                      # report only
    py src/check_duplicates.py --remove             # also write data/access_logs_deduped.csv
    py src/check_duplicates.py --file data/other.csv
The original file is never modified.
"""

import argparse
import sys
import pandas as pd

KEY_COLS = ["timestamp", "user_id", "patient_id", "action"]
LABEL_COLS = ["is_anomaly", "anomaly_type"]
MAX_EXPECTED = 5      # random coincidence gives about 0-2 in a ~20,000-event log


def check_duplicates(df, max_expected=MAX_EXPECTED):
    total = len(df)
    exact_mask = df.duplicated(keep="first")
    n_exact = int(exact_mask.sum())

    # Conflicting duplicates: same key but different labels
    key_cols = [c for c in KEY_COLS if c in df.columns]
    label_cols = [c for c in LABEL_COLS if c in df.columns]
    conflicts = df.iloc[0:0]
    if len(key_cols) == len(KEY_COLS) and label_cols:
        same_key = df[df.duplicated(subset=key_cols, keep=False)]
        if len(same_key):
            varies = same_key.groupby(key_cols)[label_cols].nunique().max(axis=1) > 1
            bad_keys = varies[varies].index
            conflicts = same_key.set_index(key_cols).loc[bad_keys].reset_index()

    print("=" * 62)
    print("DUPLICATE ROW CHECK")
    print("=" * 62)
    print(f"Total rows:                    {total}")
    print(f"Exact duplicate rows:          {n_exact}  ({100 * n_exact / total:.4f}%)")
    if len(key_cols) == len(KEY_COLS):
        print(f"Conflicting-label duplicates:  {len(conflicts)}")
    else:
        print(f"Conflicting-label check skipped (missing columns: {sorted(set(KEY_COLS) - set(df.columns))})")

    if n_exact:
        print("\nExact duplicates (first 10 shown):")
        print(df[df.duplicated(keep=False)].sort_values(list(df.columns)).head(10).to_string())
    if len(conflicts):
        print("\nConflicting duplicates (first 10 shown):")
        print(conflicts.head(10).to_string())

    if len(conflicts):
        verdict = "FAIL - the same event appears with contradictory anomaly labels. Fix the generator."
    elif n_exact > max_expected:
        verdict = (f"WARN - {n_exact} exact duplicates is more than the {max_expected} expected "
                   "from random coincidence. Check the generator.")
    else:
        verdict = (f"PASS - {n_exact} exact duplicate(s), within random coincidence. "
                   "Report it as a negligible finding." if n_exact else "PASS - no duplicates.")
    print(f"\nVerdict: {verdict}")
    print("=" * 62)
    return n_exact, len(conflicts), verdict.split(" ")[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check an access log for duplicate events.")
    parser.add_argument("--file", default="data/access_logs.csv", help="CSV to check")
    parser.add_argument("--remove", action="store_true",
                        help="write a de-duplicated copy to data/access_logs_deduped.csv")
    args = parser.parse_args()

    try:
        data = pd.read_csv(args.file)
    except FileNotFoundError:
        sys.exit(f"File not found: {args.file}\nRun access_log_generator.py first, from the project root.")

    n_exact, n_conflicts, status = check_duplicates(data)

    if args.remove:
        out = "data/access_logs_deduped.csv"
        data.drop_duplicates().to_csv(out, index=False)
        print(f"De-duplicated copy written -> {out} ({len(data) - n_exact} rows). Original unchanged.")

    sys.exit(1 if status == "FAIL" else 0)
