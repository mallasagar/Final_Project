import pandas as pd
import random
from faker import Faker

fake = Faker()

WARDS = ["General Medicine", "Cardiology", "Oncology",
         "Paediatrics", "Maternity", "ICU"]
ROLES = ["Doctor", "Nurse", "Admin"]

N_STAFF = 48                          # 8 per ward with 6 wards (keep a multiple of len(WARDS))
SHIFT_START = 9
SHIFT_LENGTH_HOURS = 8
SHIFT_END = SHIFT_START + SHIFT_LENGTH_HOURS   # 17


def generate_staff(n=N_STAFF, seed=42):
    random.seed(seed)
    staff = []
    for i in range(n):
        staff.append({
            "user_id": f"staff_{i:03d}",
            "name": fake.name(),
            "role": random.choice(ROLES),
            "assigned_ward": WARDS[i % len(WARDS)],
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
    print("\nShift check - starts:", staff_df.shift_start.unique().tolist(),
          " ends:", staff_df.shift_end.unique().tolist())
