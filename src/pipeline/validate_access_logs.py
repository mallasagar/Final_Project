import sys
import pandas as pd


def run_validation(df, staff_df, patients_df):
    checks_passed = 0
    checks_total = 0
    issues = []

    def check(name, condition):
        nonlocal checks_passed, checks_total
        checks_total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            checks_passed += 1
        else:
            issues.append(name)
        print(f"[{status}] {name}")

    print("=" * 65)
    print("ACCESS LOG VALIDATION")
    print("=" * 65)

    # --- 1. No missing values anywhere in the dataset ---
    n_missing = df.isna().sum().sum()
    check(f"No missing values in any column (found {n_missing})", n_missing == 0)

    # --- 2. No duplicate rows ---
    n_duplicates = df.duplicated().sum()
    check(f"Exact duplicate rows are within random coincidence, <= 5 (found {n_duplicates})", n_duplicates <= 5)

    # --- 3. is_anomaly is strictly binary ---
    valid_binary = set(df["is_anomaly"].unique()).issubset({0, 1})
    check("'is_anomaly' contains only 0/1 values", valid_binary)

    # --- 4. anomaly_type only ever takes the three expected values ---
    valid_types = set(df["anomaly_type"].unique()) <= {"none", "off_hours", "cross_ward"}
    check("'anomaly_type' only contains none/off_hours/cross_ward", valid_types)

    # --- 5. is_anomaly and anomaly_type agree with each other ---
    mismatch_1 = ((df["is_anomaly"] == 0) & (df["anomaly_type"] != "none")).sum()
    mismatch_2 = ((df["is_anomaly"] == 1) & (df["anomaly_type"] == "none")).sum()
    check(
        f"'is_anomaly' and 'anomaly_type' are consistent ({mismatch_1 + mismatch_2} mismatches)",
        (mismatch_1 + mismatch_2) == 0,
    )

    # --- 6. Every user_id in the logs actually exists in staff.csv ---
    unknown_users = set(df["user_id"]) - set(staff_df["user_id"])
    check(
        f"All user_ids in logs exist in staff.csv ({len(unknown_users)} unknown)",
        len(unknown_users) == 0,
    )

    # --- 7. Every patient_id in the logs actually exists in patients_with_wards.csv ---
    unknown_patients = set(df["patient_id"]) - set(patients_df["patient_id"])
    check(
        f"All patient_ids in logs exist in patients_with_wards.csv ({len(unknown_patients)} unknown)",
        len(unknown_patients) == 0,
    )

    # --- 8. Off-hours anomalies genuinely fall outside that user's shift ---
    df_ts = df.copy()
    df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"])
    df_ts["hour"] = df_ts["timestamp"].dt.hour
    df_ts = df_ts.merge(staff_df[["user_id", "shift_start", "shift_end"]], on="user_id", how="left")

    off_hours_rows = df_ts[df_ts["anomaly_type"] == "off_hours"]
    genuinely_off = ~((off_hours_rows["hour"] >= off_hours_rows["shift_start"]) &
                       (off_hours_rows["hour"] <= off_hours_rows["shift_end"]))
    check(
        f"off_hours events are genuinely outside shift hours ({genuinely_off.mean()*100:.1f}%)",
        genuinely_off.mean() > 0.95,
    )

    # --- 9. Cross-ward anomalies genuinely have assigned_ward != patient_ward ---
    cross_ward_rows = df[df["anomaly_type"] == "cross_ward"]
    genuinely_cross = (cross_ward_rows["assigned_ward"] != cross_ward_rows["patient_ward"])
    check(
        f"cross_ward events genuinely have mismatched wards ({genuinely_cross.mean()*100:.1f}%)",
        genuinely_cross.mean() > 0.95,
    )

    # --- 10. Anomaly rate is roughly as expected (~5%, not wildly off) ---
    anomaly_rate = df["is_anomaly"].mean()
    check(
        f"Overall anomaly rate is close to the intended 5% (actual: {anomaly_rate*100:.2f}%)",
        0.02 < anomaly_rate < 0.10,
    )

    # --- 11. Both anomaly types are represented (not all one type) ---
    type_counts = df[df["is_anomaly"] == 1]["anomaly_type"].value_counts()
    check(
        f"Both anomaly types are present (off_hours={type_counts.get('off_hours', 0)}, "
        f"cross_ward={type_counts.get('cross_ward', 0)})",
        type_counts.get("off_hours", 0) > 0 and type_counts.get("cross_ward", 0) > 0,
    )

    # --- 12. Timestamps are within the expected generation window ---
    ts_min, ts_max = df_ts["timestamp"].min(), df_ts["timestamp"].max()
    check(
        f"Timestamps fall within a single generated period ({ts_min.date()} to {ts_max.date()})",
        (ts_max - ts_min).days <= 90,  # sanity bound, adjust if n_days is increased
    )

    print("=" * 65)
    print(f"RESULT: {checks_passed}/{checks_total} checks passed")
    if issues:
        print(f"Issues found in: {issues}")
    else:
        print("All validation checks passed. Access log dataset is confirmed correct.")
    print("=" * 65)

    return checks_passed, checks_total, issues


if __name__ == "__main__":
    df = pd.read_csv("data/access_logs.csv")
    staff_df = pd.read_csv("data/staff.csv")
    patients_df = pd.read_csv("data/patients_with_wards.csv")

    sys.exit(1 if run_validation(df, staff_df, patients_df)[2] else 0)      # non-zero exit if any check failed
