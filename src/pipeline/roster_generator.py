"""
roster_generator.py
A staff ROSTER (data/roster.csv), generated together with the staff by staff_generator.py. For every staff member on every
day it says which ward they are rostered to. Everyone is rostered 09:00-17:00, every day, to their HOME ward unless they
are rostered to cover another ward.

Cover follows SPECIALTY (a department here is a ward). A person can only cover a ward their specialty is compatible with:
  Doctors   General physician -> Cardiology, Oncology      Cardiologist -> General Medicine, ICU      Oncologist -> General Medicine
            Paediatrician, Obstetrician and Intensivist stay on their own ward (specialised units).
  Nurses    General medical -> Cardiology, Oncology        Cardiac -> General Medicine, Oncology, ICU
            Oncology -> General Medicine, Cardiology       Critical care -> Cardiology, General Medicine
            Paediatric and Maternity nurses stay on their own ward.
  Administrators are never rostered to another ward.
Rules: a ward keeps at least 2 nurses and 1 doctor at home every day; cover goes preferentially to the compatible ward with
fewer people of that role. The compatibility lists and the cover rates (nurses 7% and doctors 8% of eligible staff-days) are
ASSUMPTIONS, set so rostered cover is about the size of the 5% legitimate cover in the main access log.

Columns: date, user_id, role, specialty, home_ward, rostered_ward, assignment_type (home / cover), reason, shift_start, shift_end

Normally written by staff_generator.py. To regenerate on its own (project root):   py src/pipeline/roster_generator.py
"""
import numpy as np
import pandas as pd

START_DATE = "2026-01-01"
N_DAYS = 90
COVER_RATE = {"Nurse": 0.07, "Doctor": 0.08, "Admin": 0.0}      # per day, for staff whose specialty can cover somewhere
MIN_AT_HOME = {"Nurse": 2, "Doctor": 1}
REASON = {"Nurse": "float cover (short-staffed ward)", "Doctor": "cross-cover (on-call support)"}

# which wards each home ward's staff may be rostered to (specialty compatibility; an assumption)
CAN_COVER = {
    "Doctor": {"General Medicine": ["Cardiology", "Oncology"], "Cardiology": ["General Medicine", "ICU"], "Oncology": ["General Medicine"],
               "Paediatrics": [], "Maternity": [], "ICU": []},
    "Nurse": {"General Medicine": ["Cardiology", "Oncology"], "Cardiology": ["General Medicine", "Oncology", "ICU"],
              "Oncology": ["General Medicine", "Cardiology"], "Paediatrics": [], "Maternity": [], "ICU": ["Cardiology", "General Medicine"]},
}


def generate_roster(staff_df, n_days=N_DAYS, seed=42, start_date=START_DATE):
    rng = np.random.default_rng(seed + 5150)
    wards = sorted(staff_df["assigned_ward"].unique())
    size = {(w, r): int(((staff_df.assigned_ward == w) & (staff_df.role == r)).sum()) for w in wards for r in ("Nurse", "Doctor")}
    spec = dict(zip(staff_df["user_id"], staff_df["specialty"])) if "specialty" in staff_df else {}
    rows = []
    for date in pd.date_range(start_date, periods=n_days):
        away = {}                                                   # user_id -> ward they are covering today
        for role in ("Nurse", "Doctor"):
            at_home = {w: size[(w, role)] for w in wards}
            members = staff_df[staff_df.role == role].sample(frac=1, random_state=int(rng.integers(1_000_000_000)))
            for _, m in members.iterrows():
                allowed = CAN_COVER[role].get(m.assigned_ward, [])
                if allowed and rng.random() < COVER_RATE[role] and at_home[m.assigned_ward] > MIN_AT_HOME[role]:
                    weights = np.array([1.0 / max(size[(w, role)], 1) for w in allowed]); weights /= weights.sum()
                    away[m.user_id] = str(rng.choice(allowed, p=weights)); at_home[m.assigned_ward] -= 1
        for _, s in staff_df.iterrows():
            target = away.get(s.user_id)
            rows.append({"date": date.date().isoformat(), "user_id": s.user_id, "role": s.role, "specialty": spec.get(s.user_id, ""), "home_ward": s.assigned_ward,
                         "rostered_ward": target or s.assigned_ward, "assignment_type": "cover" if target else "home",
                         "reason": REASON.get(s.role, "") if target else "", "shift_start": s.shift_start, "shift_end": s.shift_end})
    return pd.DataFrame(rows)


def summarise(roster):
    cover = roster[roster.assignment_type == "cover"]
    print("Cover assignments by role (staff-days):")
    t = roster.groupby("role").agg(staff_days=("user_id", "size"), cover_days=("assignment_type", lambda s: int((s == "cover").sum())))
    t["share_%"] = (100 * t.cover_days / t.staff_days).round(1); print(t.to_string())
    print("\nWho covers where: home ward (rows) -> covered ward (columns), cover staff-days:")
    print(pd.crosstab(cover.home_ward, cover.rostered_ward).to_string())
    print("\nExamples:")
    for _, r in cover.groupby("role").head(2).iterrows():
        print(f"  {r.date}: {r.specialty} {r.user_id} (home: {r.home_ward}) is rostered to {r.rostered_ward}  [{r.reason}]")


if __name__ == "__main__":
    staff = pd.read_csv("data/staff.csv"); roster = generate_roster(staff); roster.to_csv("data/roster.csv", index=False)
    print(f"Generated roster: {len(roster)} rows ({staff.shape[0]} staff x {roster.date.nunique()} days) -> data/roster.csv\n"); summarise(roster)
