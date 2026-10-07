import pandas as pd
import random

WARDS = ["General Medicine", "Cardiology", "Oncology",
         "Paediatrics", "Maternity", "ICU"]


def load_patients(synthea_csv_path="data/csv/patients.csv", seed=42):
    random.seed(seed)
    patients = pd.read_csv(synthea_csv_path)
    patients = patients.rename(columns={"Id": "patient_id"})     # Synthea's unique patient identifier
    patients["ward"] = [random.choice(WARDS) for _ in range(len(patients))]
    return patients[["patient_id", "ward"]]


if __name__ == "__main__":
    patients_df = load_patients()
    patients_df.to_csv("data/patients_with_wards.csv", index=False)
    print(f"Loaded {len(patients_df)} patients -> data/patients_with_wards.csv")
    print(patients_df.head())
    print("\nWard distribution:")
    print(patients_df["ward"].value_counts().sort_index())
