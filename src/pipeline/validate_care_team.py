"""
validate_care_team.py  -  checks data/care_team.csv (written by access_log_generator.py). Exits with an error code if any check fails.
Run from the project root:   py src/pipeline/validate_care_team.py
"""
import sys
import pandas as pd

CURRENT = 10 ** 5          # end_day values at or above this mean "still current"


def run_validation(care, staff, patients):
    """Run every check on the care-team table. Returns (passed, total, failed checks)."""
    issues, total = [], 0

    def check(name, ok):
        """Record one check result: print PASS or FAIL and keep the running totals."""
        nonlocal total
        total += 1
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok: issues.append(name)

    print("    " + "=" * 62 + "\n    CARE TEAM VALIDATION\n    " + "=" * 62)
    check("No missing values", not care.isna().any().any())
    check("Every user is a doctor or a nurse in the staff file", set(care["user_id"]) <= set(staff.loc[staff.role.isin(["Doctor", "Nurse"]), "user_id"]))
    check("Every patient exists in the patient file", set(care["patient_id"]) <= set(patients["patient_id"]))
    check("Every relationship ends on or after it starts", bool((care["end_day"] >= care["start_day"]).all()))
    # Current relationships (no end date): each patient needs exactly one primary doctor and one primary nurse.
    cur = care[care["end_day"] >= CURRENT].merge(staff[["user_id", "role"]], on="user_id").groupby(["patient_id", "role"]).size().unstack(fill_value=0)
    check(f"Every patient has exactly one current primary doctor and one current primary nurse ({len(cur)} of {patients.patient_id.nunique()} patients)",
          len(cur) == patients.patient_id.nunique() and bool((cur["Doctor"] == 1).all() and (cur["Nurse"] == 1).all()))
    # Former relationships: they exist because patients were transferred, and each stays active for 14 days afterwards.
    ended = care[care["end_day"] < CURRENT]
    check(f"Transfers happened: {len(ended)} former care relationships, each lasting 14 days after the transfer", len(ended) > 0 and bool(((ended["end_day"] - 0) >= 14).all()))
    # New relationships created by transfers: they must start inside the simulated period.
    new = care[(care["start_day"] > -CURRENT)]
    check(f"New care relationships (after transfers) start on a day inside the simulated period ({len(new)} of them)", bool((new["start_day"] >= 0).all()))
    print("    " + "=" * 62 + f"\n    RESULT: {total - len(issues)}/{total} checks passed")
    print("    " + ("All validation checks passed. The care team is confirmed correct." if not issues else f"{len(issues)} check(s) FAILED."))
    print("    " + "=" * 62)
    return total - len(issues), total, issues


if __name__ == "__main__":
    care, staff, patients = pd.read_csv("data/care_team.csv"), pd.read_csv("data/staff.csv"), pd.read_csv("data/patients_with_wards.csv")
    sys.exit(1 if run_validation(care, staff, patients)[2] else 0)
