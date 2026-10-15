"""
staff_generator.py
Synthetic clinical staff, spread evenly across wards (8 per ward).
Everyone works the same fixed shift: 09:00-17:00.

ROLE MIX follows New Zealand. The OECD (Health at a Glance 2025) reports 11.7 practising nurses and 3.7 practising
doctors per 1,000 population in New Zealand, i.e. about 3.2 nurses per doctor. With one administrator (ward clerk)
per ward (an assumption; no national ratio is available), the 48 staff are 10 doctors, 32 nurses and 6 administrators
(3.2 nurses per doctor). Four wards have 5 nurses + 2 doctors + 1 administrator; ICU and Maternity, modelled as the
more nurse-heavy wards (an illustrative choice), have 6 nurses + 1 doctor + 1 administrator.
Roles and specialties are NOT used as model features; they are descriptive. The staff roster (data/roster.csv) is generated at the end of this script.
"""

import pandas as pd
import random
from faker import Faker

fake = Faker()

WARDS = ["General Medicine", "Cardiology", "Oncology",
         "Paediatrics", "Maternity", "ICU"]
ROLES = ["Doctor", "Nurse", "Admin"]

# Each person's specialty follows their ward (a department here is a ward). Used by the roster to decide who can cover where.
SPECIALTY = {
    ("Doctor", "General Medicine"): "General physician", ("Doctor", "Cardiology"): "Cardiologist", ("Doctor", "Oncology"): "Oncologist",
    ("Doctor", "Paediatrics"): "Paediatrician", ("Doctor", "Maternity"): "Obstetrician", ("Doctor", "ICU"): "Intensivist",
    ("Nurse", "General Medicine"): "General medical nurse", ("Nurse", "Cardiology"): "Cardiac nurse", ("Nurse", "Oncology"): "Oncology nurse",
    ("Nurse", "Paediatrics"): "Paediatric nurse", ("Nurse", "Maternity"): "Maternity nurse", ("Nurse", "ICU"): "Critical care nurse",
}

# Staff per ward by role (8 per ward). Totals: 10 doctors, 32 nurses, 6 administrators.
ROLE_PLAN = {
    "General Medicine": {"Doctor": 2, "Nurse": 5, "Admin": 1},
    "Cardiology":       {"Doctor": 2, "Nurse": 5, "Admin": 1},
    "Oncology":         {"Doctor": 2, "Nurse": 5, "Admin": 1},
    "Paediatrics":      {"Doctor": 2, "Nurse": 5, "Admin": 1},
    "Maternity":        {"Doctor": 1, "Nurse": 6, "Admin": 1},
    "ICU":              {"Doctor": 1, "Nurse": 6, "Admin": 1},
}

N_STAFF = 48                          # 8 per ward with 6 wards (keep a multiple of len(WARDS))
SHIFT_START = 9
SHIFT_LENGTH_HOURS = 8
SHIFT_END = SHIFT_START + SHIFT_LENGTH_HOURS   # 17


def generate_staff(n=N_STAFF, seed=42):
    random.seed(seed)
    Faker.seed(seed)                                   # names are reproducible too
    per_ward = n // len(WARDS)
    assert all(sum(plan.values()) == per_ward for plan in ROLE_PLAN.values()), "ROLE_PLAN must give each ward exactly n / wards staff"
    queues = {ward: [role for role, k in plan.items() for role in [role] * k] for ward, plan in ROLE_PLAN.items()}
    staff = []
    for i in range(n):
        ward = WARDS[i % len(WARDS)]
        staff.append({
            "user_id": f"staff_{i:03d}",
            "name": fake.name(),
            "role": (role := queues[ward].pop(0)),     # roles come from the plan, not from chance
            "specialty": SPECIALTY.get((role, ward), "Ward administrator"),
            "assigned_ward": ward,
            "shift_start": SHIFT_START,
            "shift_end": SHIFT_END,
        })
    return pd.DataFrame(staff)


if __name__ == "__main__":
    staff_df = generate_staff()
    staff_df.to_csv("data/staff.csv", index=False)
    print(f"Generated {len(staff_df)} staff members -> data/staff.csv")
    print("\nStaff per ward:")
    print(staff_df["assigned_ward"].value_counts().sort_index())
    print("\nRoles:", staff_df["role"].value_counts().to_dict(),
          f" (nurses per doctor = {(staff_df.role == 'Nurse').sum() / (staff_df.role == 'Doctor').sum():.1f}; New Zealand about 3.2)")
    print("\nSpecialties:", staff_df["specialty"].nunique(), "different specialties")
    print("\nShift check - starts:", staff_df.shift_start.unique().tolist(),
          " ends:", staff_df.shift_end.unique().tolist())

    # The roster is generated at the same time as the staff: everyone works 09:00-17:00 every day, and cover follows specialty.
    from roster_generator import generate_roster, summarise
    roster_df = generate_roster(staff_df)
    roster_df.to_csv("data/roster.csv", index=False)
    print(f"\nRoster generated with the staff: {len(roster_df)} rows -> data/roster.csv\n")
    summarise(roster_df)
