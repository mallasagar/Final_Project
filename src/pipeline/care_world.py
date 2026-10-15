"""
care_world.py
The simulated hospital behind the access log: a roster, care relationships over time, and (optionally) devices.

Roster (roster_generator.py)  Everyone works 09:00-17:00; people may be rostered to cover a specialty-compatible ward.
Continuity of care
  * Every patient has a care team: a primary doctor and a primary nurse on their ward.
  * FOLLOW-UPS: a patient returns every 14-35 days and sees the SAME primary doctor again.
  * TRANSFERS: about 5 patients a day move to another ward. The former doctor and nurse keep a care relationship for 14 days and may
    legitimately look the patient up again, even though the patient is now on another ward.
  * Some ad-hoc access has no history behind it (genuine emergencies and consults): 1 - HISTORY_SHARE of it (an assumption).
  The care team is a lookup table (data/care_team.csv), like the roster. 'care_relationship' says whether the user was on the patient's
  care team on the day of the access. It is a plain lookup, never a label.
Anomalies (drawn independently of the roster and the care team)
  * off-hours: 00:00-05:59 or 20:00-23:59, own-ward patient.    * cross-ward: inside the shift, on a ward that is neither the user's home
  ward nor their rostered ward.
Devices (used only by the EXTENSION in src/extension/, because stolen credentials are a THIRD scenario outside the proposal)
  * Registered devices and ward workstations, VPN, external locations; with stolen_rate > 0, stealth stolen-credential events appear:
    inside the shift, own ward, only the device and location reveal them.
Patients can change ward, so 'patient_ward' is the ward at the time of the access.
"""
import sys
from collections import defaultdict
from pathlib import Path


def _add_script_folders():
    """Make every script folder importable, so each script can import the others whichever folder it is run from."""
    here = Path(__file__).resolve().parent
    skip = {"venv", "__pycache__", "data", "results", "models", "logs"}
    for base in (here, here.parent):
        for p in [base] + [d for d in base.iterdir() if d.is_dir() and d.name not in skip and not d.name.startswith(".")]:
            if str(p) not in sys.path:
                sys.path.append(str(p))


_add_script_folders()

import numpy as np
import pandas as pd

from roster_generator import generate_roster, START_DATE

EVENTS_PER_DAY = 240
ANOMALY_RATE = 0.05
STOLEN_RATE = 0.0                # stolen-credential events: 0 in the main design; the extension sets 0.015
OUTSIDE_SHIFT_RATE = 0.10
OFF_HOURS_MARGIN = 3
ADHOC_RATE = 0.015               # ASSUMPTION: share of doctor/nurse events that are legitimate access NOT on the roster
COVER_DAY_ON_ROSTERED_WARD = 0.90
HISTORY_SHARE = 0.80             # ASSUMPTION: share of ad-hoc access that has a care relationship behind it
TRANSFER_PER_DAY = 5             # ASSUMPTION: patients transferred between wards per day
FOLLOWUP_DAYS = (14, 36)         # ASSUMPTION: days between a patient's follow-up visits (14 to 35)
FORMER_TEAM_DAYS = 14            # a former team keeps a reason to look again for this long
NURSE_PANEL_SHARE = 0.6
WORKSTATION_SHARE, UNREGISTERED_LEGIT, VPN_DOCTOR, EXTERNAL_LEGIT = 0.15, 0.01, 0.03, 0.003
STOLEN_EXTERNAL = 0.80
INF = 10 ** 6


def generate_care_world(patients_df, staff_df, n_days, seed, history_share=HISTORY_SHARE, adhoc_rate=ADHOC_RATE, anomaly_rate=ANOMALY_RATE,
                        stolen_rate=STOLEN_RATE, events_per_day=EVENTS_PER_DAY, outside_rate=OUTSIDE_SHIFT_RATE, margin=OFF_HOURS_MARGIN,
                        unregistered_rate=UNREGISTERED_LEGIT, roster_df=None):
    """Simulate the hospital for n_days. Returns (the access log, the roster, the care-team table). The seed fully determines the result; the keyword settings are the assumptions the stress tests vary."""
    # One random stream for the whole simulated hospital, so a seed fully determines the dataset.
    rng = np.random.default_rng(seed)
    staff = staff_df.reset_index(drop=True); n_staff = len(staff)
    roster = roster_df if roster_df is not None else generate_roster(staff, n_days=n_days, seed=seed)
    R = roster.pivot(index="date", columns="user_id", values="rostered_ward").reindex(columns=staff["user_id"]).to_numpy()
    wards = sorted(staff["assigned_ward"].unique())
    rnd = lambda lst: lst[int(rng.integers(len(lst)))]

    # ---- hospital state: patients, care teams, follow-up schedule ----
    cur_ward = dict(zip(patients_df["patient_id"], patients_df["ward"]))
    ward_pat = {w: list(patients_df.loc[patients_df["ward"] == w, "patient_id"]) for w in wards}
    docs = {w: list(staff.loc[(staff.assigned_ward == w) & (staff.role == "Doctor"), "user_id"]) for w in wards}
    nurs = {w: list(staff.loc[(staff.assigned_ward == w) & (staff.role == "Nurse"), "user_id"]) for w in wards}
    all_docs = [u for w in wards for u in docs[w]]
    pdoc, pnur, rel, panel, former = {}, {}, {}, defaultdict(list), defaultdict(list)
    for w in wards:
        for k, pid in enumerate(ward_pat[w]):
            pdoc[pid], pnur[pid] = docs[w][k % len(docs[w])], nurs[w][k % len(nurs[w])]
            rel[(pid, pdoc[pid])], rel[(pid, pnur[pid])] = [-INF, INF], [-INF, INF]
            panel[pdoc[pid]].append(pid); panel[pnur[pid]].append(pid)
    # Each patient's first follow-up is due on a random day in the first five weeks.
    next_due = {pid: int(rng.integers(0, FOLLOWUP_DAYS[1])) for pid in cur_ward}
    all_pids = list(cur_ward)

    # ---- who, when, what kind of event (vectorised) ----
    n = n_days * events_per_day
    day = np.repeat(np.arange(n_days), events_per_day)
    idx = rng.integers(0, n_staff, n)
    user, role, home = staff["user_id"].to_numpy()[idx], staff["role"].to_numpy()[idx], staff["assigned_ward"].to_numpy()[idx]
    start, end = staff["shift_start"].to_numpy()[idx].astype(int), staff["shift_end"].to_numpy()[idx].astype(int)
    rost = R[day, idx]
    is_anom = rng.random(n) < anomaly_rate
    atype = np.where(is_anom, np.where(rng.random(n) < 0.5, "off_hours", "cross_ward"), "none")
    stolen = (~is_anom) & (rng.random(n) < stolen_rate)
    atype = np.where(stolen, "stolen_credentials", atype)
    normal, off = atype == "none", atype == "off_hours"
    hour = rng.integers(start, end)
    outside = normal & (rng.random(n) < outside_rate)
    hour = np.where(outside, np.where(rng.random(n) < 0.5, start - 1, end), hour)
    before, after = np.maximum(start - margin, 0), np.maximum(24 - (end + margin), 0)
    if np.any(before + after == 0):
        raise ValueError("Shift leaves no room for off-hours anomalies; reduce the margin.")
    k = rng.integers(0, np.maximum(before + after, 1))
    hour = np.where(off, np.where(k < before, k, end + margin + (k - before)), hour)

    # ---- which patient: sequential, because transfers and follow-ups change the hospital over time ----
    pid_arr, pw_arr = np.empty(n, dtype=object), np.empty(n, dtype=object)
    cur_day, due = -1, {}
    # Events are processed in time order, because transfers and follow-ups change the hospital as the days pass.
    for i in range(n):
        d = int(day[i])
        if d != cur_day:
            cur_day = d
            for _ in range(int(rng.poisson(TRANSFER_PER_DAY))):                               # patients moving between wards
                pid = rnd(all_pids); old = cur_ward[pid]; new = rnd([w for w in wards if w != old])
                ward_pat[old].remove(pid); ward_pat[new].append(pid); cur_ward[pid] = new
                od, on = pdoc[pid], pnur[pid]
                rel[(pid, od)][1] = rel[(pid, on)][1] = d + FORMER_TEAM_DAYS
                former[od].append((pid, d + FORMER_TEAM_DAYS)); former[on].append((pid, d + FORMER_TEAM_DAYS))
                panel[od].remove(pid); panel[on].remove(pid)
                nd, nn = rnd(docs[new]), rnd(nurs[new]); pdoc[pid], pnur[pid] = nd, nn
                rel[(pid, nd)], rel[(pid, nn)] = [d, INF], [d, INF]; panel[nd].append(pid); panel[nn].append(pid)
                next_due[pid] = d + int(rng.integers(*FOLLOWUP_DAYS))
            due = {doc: [p for p in panel[doc] if pdoc[p] == doc and next_due[p] <= d] for doc in all_docs}   # follow-ups due today
        u, r, h, ro, t = user[i], role[i], home[i], rost[i], atype[i]
        if t in ("off_hours", "stolen_credentials"):
            pid = rnd(ward_pat[h])
        elif t == "cross_ward":
            pid = rnd(ward_pat[rnd([w for w in wards if w != h and w != ro])])
        elif ro != h and rng.random() < COVER_DAY_ON_ROSTERED_WARD:
            pid = rnd(ward_pat[ro])                                                            # rostered cover
        elif r != "Admin" and rng.random() < adhoc_rate:                                      # ad-hoc access
            fl = [p for p, e in former[u] if e >= d and cur_ward[p] != h]
            pid = rnd(fl) if (fl and rng.random() < history_share) else rnd(ward_pat[rnd([w for w in wards if w != h and w != ro])])
        elif r == "Doctor" and due.get(u):
            pid = due[u].pop(int(rng.integers(len(due[u])))); next_due[pid] = d + int(rng.integers(*FOLLOWUP_DAYS))   # FOLLOW-UP with the same doctor
        elif r == "Doctor" and panel[u] and rng.random() < 0.5:
            pid = rnd(panel[u])
        elif r == "Nurse" and panel[u] and rng.random() < NURSE_PANEL_SHARE:
            pid = rnd(panel[u])
        else:
            pid = rnd(ward_pat[h])
        pid_arr[i], pw_arr[i] = pid, cur_ward[pid]

    # ---- device and location (only the extension uses them) ----
    # Device and location columns (not used by the main design, which drops them; see access_log_generator.py).
    # This block is kept, and not deleted, because it consumes random numbers: removing it would shift the random stream and slightly change every result.
    u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    device = np.where(u1 < WORKSTATION_SHARE, "WS-" + home.astype(str), "DEV-" + user.astype(str))
    device = np.where(normal & (u2 < unregistered_rate), "UNK-NEW-" + user.astype(str), device)
    device = np.where(stolen, np.char.add("UNK-ATTACK-", rng.integers(0, 10 ** 6, n).astype(str)), device)
    location = np.where((role == "Doctor") & (u3 < VPN_DOCTOR), "vpn", "onsite")
    location = np.where(normal & (rng.random(n) < EXTERNAL_LEGIT), "external", location)
    location = np.where(stolen, np.where(rng.random(n) < STOLEN_EXTERNAL, "external", "onsite"), location)

    ts = (pd.Timestamp(START_DATE) + pd.to_timedelta(day, "D") + pd.to_timedelta(hour, "h") + pd.to_timedelta(rng.integers(0, 60, n), "m"))
    logs = pd.DataFrame({"timestamp": ts, "user_id": user, "role": role, "assigned_ward": home, "rostered_ward": rost, "patient_id": pid_arr,
                         "patient_ward": pw_arr, "action": rng.choice(["read", "read", "read", "write", "export"], n), "device_id": device, "location": location,
                         "is_anomaly": (is_anom | stolen).astype(int), "anomaly_type": atype}).sort_values("timestamp").reset_index(drop=True)
    care_team = pd.DataFrame([(p, uu, s, e) for (p, uu), (s, e) in rel.items()], columns=["patient_id", "user_id", "start_day", "end_day"])
    return logs, roster, care_team


def add_care_lookup(logs, care_team):
    """care_relationship = 1 if the user was on the patient's care team on the day of the access (a lookup; no label is used)."""
    day = (pd.to_datetime(logs["timestamp"]).dt.normalize() - pd.Timestamp(START_DATE)).dt.days
    m = logs[["user_id", "patient_id"]].assign(day=day.to_numpy()).reset_index().merge(care_team, on=["user_id", "patient_id"], how="left")
    m["hit"] = (m["start_day"] <= m["day"]) & (m["day"] <= m["end_day"])
    out = logs.copy(); out["care_relationship"] = m.groupby("index")["hit"].any().reindex(logs.index).fillna(False).astype(int).to_numpy()
    return out
