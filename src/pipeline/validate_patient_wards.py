import sys
import pandas as pd

VALID_WARDS = {"General Medicine", "Cardiology", "Oncology", "Paediatrics", "Maternity", "ICU"}


def run_validation(df, raw_patients_df):
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
    print("PATIENT-WARD MAPPING VALIDATION")
    print("=" * 65)

    # --- 1. No missing values ---
    n_missing = df.isna().sum().sum()
    check(f"No missing values in any column (found {n_missing})", n_missing == 0)

    # --- 2. patient_id is unique for every row (one ward per patient) ---
    n_duplicate_ids = df["patient_id"].duplicated().sum()
    check(f"'patient_id' values are all unique ({n_duplicate_ids} duplicates)", n_duplicate_ids == 0)

    # --- 3. No duplicate rows overall ---
    n_duplicate_rows = df.duplicated().sum()
    check(f"No exact duplicate rows (found {n_duplicate_rows})", n_duplicate_rows == 0)

    # --- 4. ward only contains the five expected values ---
    valid_wards = set(df["ward"].unique()) <= VALID_WARDS
    check(f"'ward' only contains {VALID_WARDS}", valid_wards)

    # --- 5. Every ward has at least one patient assigned ---
    ward_counts = df["ward"].value_counts()
    all_wards_used = VALID_WARDS.issubset(set(ward_counts.index))
    check(f"Every ward has at least one patient ({ward_counts.to_dict()})", all_wards_used)

    # --- 6. Row count matches the raw Synthea patient count (no patients lost or added) ---
    counts_match = len(df) == len(raw_patients_df)
    check(
        f"Patient count matches raw Synthea file ({len(df)} vs {len(raw_patients_df)})",
        counts_match,
    )

    # --- 7. Every patient_id in the mapping exists in the original Synthea data ---
    raw_ids = set(raw_patients_df["Id"]) if "Id" in raw_patients_df.columns else set(raw_patients_df["patient_id"])
    mapped_ids = set(df["patient_id"])
    unmatched = mapped_ids - raw_ids
    check(
        f"All mapped patient_ids trace back to the original Synthea file ({len(unmatched)} unmatched)",
        len(unmatched) == 0,
    )

    # --- 8. No Synthea patient was silently dropped during mapping ---
    dropped = raw_ids - mapped_ids
    check(
        f"No Synthea patients were dropped during ward assignment ({len(dropped)} dropped)",
        len(dropped) == 0,
    )

    # --- 9. Ward distribution is not wildly unbalanced (rough sanity check) ---
    min_share = ward_counts.min() / len(df)
    max_share = ward_counts.max() / len(df)
    reasonably_balanced = min_share > 0.05  # no ward with less than ~5% of patients
    check(
        f"Ward distribution is not extremely skewed (smallest ward: {min_share*100:.1f}% of patients)",
        reasonably_balanced,
    )

    print("=" * 65)
    print(f"RESULT: {checks_passed}/{checks_total} checks passed")
    if issues:
        print(f"Issues found in: {issues}")
    else:
        print("All validation checks passed. Patient-ward mapping is confirmed correct.")
    print("=" * 65)

    return checks_passed, checks_total, issues


if __name__ == "__main__":
    df = pd.read_csv("data/patients_with_wards.csv")
    raw_patients_df = pd.read_csv("data/csv/patients.csv")

    sys.exit(1 if run_validation(df, raw_patients_df)[2] else 0)      # non-zero exit if any check failed
