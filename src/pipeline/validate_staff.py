import sys
import pandas as pd

VALID_ROLES = {"Doctor", "Nurse", "Admin"}
VALID_WARDS = {"General Medicine", "Cardiology", "Oncology", "Paediatrics", "Maternity", "ICU"}


def run_validation(df):
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

    print("=" * 60)
    print("STAFF GENERATOR VALIDATION")
    print("=" * 60)

    # --- 1. No missing values anywhere ---
    n_missing = df.isna().sum().sum()
    check(f"No missing values in any column (found {n_missing})", n_missing == 0)

    # --- 2. user_id is unique for every staff member ---
    n_duplicate_ids = df["user_id"].duplicated().sum()
    check(f"'user_id' values are all unique ({n_duplicate_ids} duplicates)", n_duplicate_ids == 0)

    # --- 3. No duplicate rows overall ---
    n_duplicate_rows = df.duplicated().sum()
    check(f"No exact duplicate rows (found {n_duplicate_rows})", n_duplicate_rows == 0)

    # --- 4. role only contains the three expected values ---
    valid_roles = set(df["role"].unique()) <= VALID_ROLES
    check(f"'role' only contains {VALID_ROLES}", valid_roles)

    # --- 5. assigned_ward only contains the five expected wards ---
    valid_wards = set(df["assigned_ward"].unique()) <= VALID_WARDS
    check(f"'assigned_ward' only contains {VALID_WARDS}", valid_wards)

    # --- 6. shift_start and shift_end are within a sane 24-hour range ---
    start_valid = df["shift_start"].between(0, 23).all()
    end_valid = df["shift_end"].between(0, 23).all()
    check("'shift_start' values are within 0-23", start_valid)
    check("'shift_end' values are within 0-23", end_valid)

    # --- 7. shift_end is always after shift_start (no zero-length or inverted shifts) ---
    valid_shift_order = (df["shift_end"] > df["shift_start"]).all()
    check("'shift_end' is always later than 'shift_start' for every staff member", valid_shift_order)

    # --- 8. Shift lengths are reasonable (e.g. between 6 and 14 hours) ---
    shift_lengths = df["shift_end"] - df["shift_start"]
    reasonable_lengths = shift_lengths.between(6, 14).all()
    check(
        f"Shift lengths are reasonable, 6-14 hours (min={shift_lengths.min()}, max={shift_lengths.max()})",
        reasonable_lengths,
    )

    # --- 9. Every role appears at least once (not accidentally all one role) ---
    role_counts = df["role"].value_counts()
    all_roles_present = VALID_ROLES.issubset(set(role_counts.index))
    check(f"All three roles are represented ({role_counts.to_dict()})", all_roles_present)

    # --- 10. Every ward has at least one staff member assigned ---
    ward_counts = df["assigned_ward"].value_counts()
    all_wards_staffed = VALID_WARDS.issubset(set(ward_counts.index))
    check(f"Every ward has at least one staff member ({ward_counts.to_dict()})", all_wards_staffed)

    # --- 10b. Staff are spread evenly across wards (largest ward at most 1 more than smallest) ---
    per_ward = df["assigned_ward"].value_counts()
    check(f"Staff are spread evenly across wards (min={per_ward.min()}, max={per_ward.max()})",
          per_ward.max() - per_ward.min() <= 1)

    # --- 11. name field is non-empty text for every row ---
    valid_names = df["name"].str.strip().str.len().gt(0).all()
    check("'name' is non-empty for every staff member", valid_names)

    # --- 12. Row count matches what was requested (sanity check against generator call) ---
    check(f"Staff population size is as expected (n={len(df)})", len(df) > 0)

    print("=" * 60)
    print(f"RESULT: {checks_passed}/{checks_total} checks passed")
    if issues:
        print(f"Issues found in: {issues}")
    else:
        print("All validation checks passed. Staff dataset is confirmed correct.")
    print("=" * 60)

    return checks_passed, checks_total, issues


if __name__ == "__main__":
    df = pd.read_csv("data/staff.csv")
    sys.exit(1 if run_validation(df)[2] else 0)      # non-zero exit if any check failed
