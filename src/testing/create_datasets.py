"""
create_datasets.py
STEP 1 of 3 - CREATE datasets. Each one gets its own folder under testing/datasets/ containing:
    access_logs.csv   the raw simulated access log (with the true anomaly labels)
    config.json       exactly how it was made (seed, days, noise, anomaly rate ...)

Two ways to create them:
    py src/testing/create_datasets.py --n 10                 # 10 datasets, same settings, different seeds
    py src/testing/create_datasets.py --profiles             # one dataset per stress profile (noisier, subtler, ...)
    py src/testing/create_datasets.py --profiles --repeats 3 # three datasets per profile
Add --clean to delete the previous testing/ datasets and comparison first.
"""
import argparse
import json
import shutil
from datetime import datetime

import pandas as pd

import _common as C

C.add_script_folders()
import access_log_generator as gen

DEFAULTS = dict(days=30, anomaly_rate=0.05, outside=0.10, cover=0.015, margin=3)      # "cover" = unrostered (ad-hoc) legitimate cross-ward access rate
PROFILES = {
    "baseline": {},
    "more_cover_noise": {"cover": 0.03},                  # 3% of doctor/nurse events are unrostered cross-ward access (design 1.5%)
    "heavy_cover_noise": {"cover": 0.06},
    "more_outside_noise": {"outside": 0.30},              # 30% of legitimate events one hour outside the shift (design 10%)
    "subtle_anomalies": {"margin": 0},                    # off-hours anomalies as close to the shift as legitimate lateness
    "higher_anomaly_rate": {"anomaly_rate": 0.10},
    "lower_anomaly_rate": {"anomaly_rate": 0.02},
    "short_log_7_days": {"days": 7},
}


def build(cfg, seed, staff, patients):
    """Generate one raw access log with the given settings, then restore the generator's design values."""
    gen.OUTSIDE_SHIFT_RATE, gen.ADHOC_RATE, gen.OFF_HOURS_MARGIN = cfg["outside"], cfg["cover"], cfg["margin"]
    try:
        return gen.generate_access_logs(patients, staff, n_days=cfg["days"], events_per_day=gen.EVENTS_PER_DAY,
                                        anomaly_rate=cfg["anomaly_rate"], seed=seed)
    finally:
        gen.OUTSIDE_SHIFT_RATE, gen.ADHOC_RATE, gen.OFF_HOURS_MARGIN = 0.10, 0.015, 3


def main():
    ap = argparse.ArgumentParser(description="Create datasets, one folder each, under testing/datasets/.")
    ap.add_argument("--n", type=int, default=10, help="number of datasets when not using --profiles")
    ap.add_argument("--profiles", action="store_true", help="one dataset per stress profile")
    ap.add_argument("--repeats", type=int, default=1, help="datasets per profile (with --profiles)")
    ap.add_argument("--start-seed", type=int, default=40000, help="first seed (use a range you never used before)")
    ap.add_argument("--days", type=int, default=None, help="days per dataset (default 30)")
    ap.add_argument("--clean", action="store_true", help="delete earlier testing/datasets and testing/comparison first")
    a = ap.parse_args()

    if a.clean:
        for d in (C.DATASETS, C.COMPARISON):
            if d.exists():
                shutil.rmtree(d); print(f"removed {d}")
    C.DATASETS.mkdir(parents=True, exist_ok=True)
    staff, patients = pd.read_csv(C.ROOT / "data" / "staff.csv"), pd.read_csv(C.ROOT / "data" / "patients_with_wards.csv")

    plan = []                                              # (profile, overrides)
    if a.profiles:
        plan = [(name, ov) for name, ov in PROFILES.items() for _ in range(a.repeats)]
    else:
        plan = [("baseline", {}) for _ in range(a.n)]

    existing = len([d for d in C.DATASETS.iterdir() if d.is_dir()])
    print(f"Creating {len(plan)} dataset(s) in {C.DATASETS}\n")
    print(f"{'folder':<42}{'events':>8}{'anomalies':>11}  settings that differ from the design")
    for i, (profile, ov) in enumerate(plan):
        seed = a.start_seed + existing + i
        cfg = {**DEFAULTS, **ov}
        if a.days: cfg["days"] = a.days if "days" not in ov else cfg["days"]
        name = f"{existing + i + 1:02d}_{profile}_seed{seed}" if a.profiles else f"{existing + i + 1:02d}_seed{seed}"
        folder = C.DATASETS / name; folder.mkdir(parents=True, exist_ok=False)
        logs = build(cfg, seed, staff, patients)
        logs.to_csv(folder / "access_logs.csv", index=False)
        with open(folder / "config.json", "w") as f:
            json.dump({"name": name, "profile": profile, "seed": seed, **cfg, "created": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
        diff = ", ".join(f"{k}={v}" for k, v in ov.items()) or "none (design settings)"
        print(f"{name:<42}{len(logs):>8}{int(logs.is_anomaly.sum()):>11}  {diff}")
    print("\nNext: py src/testing/test_datasets.py")


if __name__ == "__main__":
    main()
