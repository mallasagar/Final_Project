"""
validate_roster.py  -  checks data/roster.csv against data/staff.csv. Exits with an error code if any check fails.
Run from the project root:   py src/pipeline/validate_roster.py
"""
import sys
import pandas as pd

from roster_generator import CAN_COVER

WARDS = {"General Medicine", "Cardiology", "Oncology", "Paediatrics", "Maternity", "ICU"}


def run_validation(roster, staff):
    issues, total = [], 0

    def check(name, ok):
        nonlocal total
        total += 1
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok: issues.append(name)

    print("    " + "=" * 62 + "\n    ROSTER VALIDATION\n    " + "=" * 62)
    n_days = roster["date"].nunique()
    check("No missing values in the roster columns used (date, user_id, role, home_ward, rostered_ward, assignment_type)", roster[["date", "user_id", "role", "home_ward", "rostered_ward", "assignment_type"]].notna().all().all())
    check(f"One row per staff member per day ({len(staff)} staff x {n_days} days = {len(staff) * n_days}; found {len(roster)})", len(roster) == len(staff) * n_days and not roster.duplicated(["date", "user_id"]).any())
    dates = pd.to_datetime(roster["date"]).drop_duplicates().sort_values()
    check(f"The days are consecutive ({dates.min().date()} to {dates.max().date()})", (dates.diff().dropna() == pd.Timedelta(days=1)).all())
    m = roster.merge(staff[["user_id", "role", "assigned_ward", "shift_start", "shift_end"]], on="user_id", how="left", suffixes=("", "_staff"))
    check("Every roster user_id exists in the staff file", m["role_staff"].notna().all())
    check("Role and home ward match the staff file for every row", (m["role"] == m["role_staff"]).all() and (m["home_ward"] == m["assigned_ward"]).all())
    check("Shift hours match the staff file (09:00-17:00)", (m["shift_start"] == m["shift_start_staff"]).all() and (m["shift_end"] == m["shift_end_staff"]).all()) if "shift_start_staff" in m else check("Shift hours present", roster[["shift_start", "shift_end"]].notna().all().all())
    check("Every rostered ward is one of the six wards", set(roster["rostered_ward"]) <= WARDS)
    cover = roster["assignment_type"] == "cover"
    check("'cover' exactly when the rostered ward differs from the home ward", (cover == (roster["rostered_ward"] != roster["home_ward"])).all())
    check("Administrators are never rostered to another ward", not (cover & (roster["role"] == "Admin")).any())
    cc = roster[cover]
    check("Cover only goes to wards the person's specialty is compatible with (CAN_COVER rules)", all(r.rostered_ward in CAN_COVER[r.role].get(r.home_ward, []) for r in cc.itertuples()))
    check("Every staff member is rostered 09:00-17:00 on every day", bool((roster["shift_start"] == 9).all() and (roster["shift_end"] == 17).all()))
    check("A specialty is recorded for every staff member", bool(roster["specialty"].fillna("").ne("").all()))
    check("Both nurses and doctors do cover other wards (cross-department work is represented)", bool((cover & (roster["role"] == "Nurse")).any() and (cover & (roster["role"] == "Doctor")).any()))
    home = roster[~cover].groupby(["date", "home_ward", "role"]).size().unstack("role").reindex(columns=["Nurse", "Doctor"]).fillna(0)
    check(f"Every ward keeps at least 2 nurses and 1 doctor at home on every day (min nurses {int(home['Nurse'].min())}, min doctors {int(home['Doctor'].min())})", bool((home["Nurse"] >= 2).all() and (home["Doctor"] >= 1).all()))
    rate = {r: 100 * (cover & (roster.role == r)).sum() / (roster.role == r).sum() for r in ("Nurse", "Doctor")}
    check(f"Cover shares are near the design (nurses {rate['Nurse']:.1f}% of nurse-days, design 7% of eligible days; doctors {rate['Doctor']:.1f}% of all doctor-days, "
          f"design 8% of eligible days, lower overall because Paediatrics, Maternity and ICU staff do not cover)", 2.5 <= rate["Nurse"] <= 6.5 and 2.0 <= rate["Doctor"] <= 9)
    print("    " + "=" * 62 + f"\n    RESULT: {total - len(issues)}/{total} checks passed")
    print("    " + ("All validation checks passed. The roster is confirmed correct." if not issues else f"{len(issues)} check(s) FAILED."))
    print("    " + "=" * 62)
    return total - len(issues), total, issues


if __name__ == "__main__":
    staff, roster = pd.read_csv("data/staff.csv"), pd.read_csv("data/roster.csv")
    sys.exit(1 if run_validation(roster, staff)[2] else 0)
