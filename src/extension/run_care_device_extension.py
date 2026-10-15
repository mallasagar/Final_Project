"""
run_care_device_extension.py   (EXTENSION - writes only to extension_care/; the main results are never touched)
What do DEVICE and LOCATION add on top of the main design (roster + care team)?   [stolen credentials are a THIRD scenario, NOT in the proposal]

  Device features   'device_known' (a registered device or the ward workstation) and 'location_trusted' (on-site or a registered VPN).
  Stolen credentials  an extra 1.5% of events, built to look normal in time and ward (inside the shift, own-ward patient): only the device and
                      location reveal them (unknown device; external location 80%, on-site 20%). That they are detectable only through the device
                      is true BY DESIGN of this scenario, and a device check is a standard Basic Zero Trust policy.
Models (Isolation Forest, 200 trees; flags chosen by a review budget fixed in advance)   E0 main model    E2 main + device/location
Rules   P0 Basic policy (main: hour OR unjustified access)      P2 + device/location (unknown device or untrusted location)
Train: one 90-day world (seed 42, with stolen credentials). Test: 20 fresh 30-day worlds (seeds 90000-90019). Sensitivity: how much legitimate
noise there is (new laptops, travelling): 0.3%, 1% (design) and 3% of events on an unregistered device.

Run from the project root:   py src/extension/run_care_device_extension.py
Outputs (extension_care/): care_device_results.csv, care_device_summary.csv, care_device_false_alarms.csv, care_device_sensitivity.csv, fig_care_device.png / .pdf
"""
import sys
from pathlib import Path


def _add_script_folders():
    here = Path(__file__).resolve().parent
    skip = {"venv", "__pycache__", "data", "results", "models", "logs"}
    for base in (here, here.parent):
        for p in [base] + [d for d in base.iterdir() if d.is_dir() and d.name not in skip and not d.name.startswith(".")]:
            if str(p) not in sys.path:
                sys.path.append(str(p))


_add_script_folders()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

import access_log_generator as gen
from care_world import generate_care_world, add_care_lookup, UNREGISTERED_LEGIT
from feature_engineering import engineer_features, FEATURE_COLS
from run_experiment import DEFAULT_WINDOW

OUT = Path("extension_care")
EXT_STOLEN = 0.015
TEST_SEEDS = list(range(90000, 90020)); SENS_RATES, SENS_SEEDS = [0.003, UNREGISTERED_LEGIT, 0.03], TEST_SEEDS[:10]
FEATS = {"E0 main model": FEATURE_COLS, "E2 + device/location": FEATURE_COLS + ["device_known", "location_trusted"]}
SCEN = ["off_hours", "cross_ward", "stolen_credentials"]


def make(staff, patients, days, seed, unreg=UNREGISTERED_LEGIT):
    logs, roster, care = generate_care_world(patients, staff, days, seed, history_share=gen.HISTORY_SHARE, adhoc_rate=gen.ADHOC_RATE, stolen_rate=EXT_STOLEN, unregistered_rate=unreg)
    df = engineer_features(add_care_lookup(logs, care), staff)[0]
    df["device_known"] = ((df["device_id"] == "DEV-" + df["user_id"]) | (df["device_id"] == "WS-" + df["assigned_ward"]) | (df["device_id"] == "WS-" + df["rostered_ward"])).astype(int)
    df["location_trusted"] = (df["location"] != "external").astype(int)
    return df, logs, care


def policies(d):
    h = ((d["hour"] < DEFAULT_WINDOW[0]) | (d["hour"] > DEFAULT_WINDOW[1])).to_numpy(); un = (d["justified_access"] == 0).to_numpy()
    dev = ((d["device_known"] == 0) | (d["location_trusted"] == 0)).to_numpy()
    return {"P0 Basic policy (main)": h | un, "P2 + device/location": h | un | dev}


def topk(score, frac):
    f = np.zeros(len(score), bool); f[np.argsort(-score, kind="stable")[:int(round(frac * len(score)))]] = True; return f


def metr(d, flag):
    y, ct = d["is_anomaly"].to_numpy(), d["anomaly_type"].to_numpy(); tp, fp = (flag & (y == 1)).sum(), (flag & (y == 0)).sum(); p = tp / max(tp + fp, 1); r = tp / y.sum()
    m = {"detection_%": 100 * r, "false_pos_%": 100 * fp / (y == 0).sum(), "precision_%": 100 * p, "F1_%": 100 * 2 * p * r / max(p + r, 1e-9)}
    m.update({s + "_%": 100 * flag[ct == s].mean() for s in SCEN}); return m


def evaluate(d, models):
    out, pol = [], policies(d); vol = float(pol["P2 + device/location"].mean())
    for name, f in pol.items(): out.append({"condition": name, "budget": "fixed rule", **metr(d, f)})
    for name, cols in FEATS.items():
        sc = -models[name].decision_function(d[cols])
        for lab, frac in [("5% review budget", 0.05), ("7% review budget", 0.07), ("same alert volume as P2 (about 7%)", vol)]:
            out.append({"condition": name, "budget": lab, **metr(d, topk(sc, frac)), "roc_auc": roc_auc_score(d["is_anomaly"], sc)})
    return out


def train(staff, patients, unreg=UNREGISTERED_LEGIT):
    tr, logs, care = make(staff, patients, 90, 42, unreg)
    return tr, {n: IsolationForest(n_estimators=200, contamination=0.05, random_state=42).fit(tr[c]) for n, c in FEATS.items()}


if __name__ == "__main__":
    staff, patients = pd.read_csv("data/staff.csv"), pd.read_csv("data/patients_with_wards.csv")
    OUT.mkdir(exist_ok=True)
    print("=" * 80 + "\nDEVICE AND LOCATION EXTENSION on top of the main design (writes only to extension_care/)\n" + "=" * 80)
    tr, models = train(staff, patients); st = tr[tr.anomaly_type == "stolen_credentials"]; lg = tr[tr.is_anomaly == 0]
    checks = [("stolen credentials are about 1.5% of events", 0.01 <= len(st) / len(tr) <= 0.02),
              ("stolen credentials: inside the shift, own ward, unknown device", bool((st.hours_beyond_shift == 0).all() and (st.authorised_ward == 1).all() and (st.device_known == 0).all())),
              ("legitimate events mostly have a known device and a trusted location", bool(lg.device_known.mean() > 0.97 and lg.location_trusted.mean() > 0.99)),
              ("no label column is a model feature", not ({"is_anomaly", "anomaly_type"} & set(sum(FEATS.values(), []))))]
    print("  Data checks on the 90-day training world:")
    for n, ok in checks: print(f"    [{'PASS' if ok else 'FAIL'}] {n}")
    if not all(ok for _, ok in checks): raise SystemExit("A data check failed.")
    rows, tests = [], []
    for s in TEST_SEEDS:
        d, _, _ = make(staff, patients, 30, s); tests.append(d)
        for r in evaluate(d, models): rows.append({"seed": s, **r})
    res = pd.DataFrame(rows); res.round(4).to_csv(OUT / "care_device_results.csv", index=False)
    mc = ["detection_%", "off_hours_%", "cross_ward_%", "stolen_credentials_%", "false_pos_%", "precision_%", "F1_%", "roc_auc"]
    summ = res.groupby(["budget", "condition"], sort=False)[mc].mean().round(2); summ.to_csv(OUT / "care_device_summary.csv"); pd.set_option("display.width", 200)
    print(f"\n=== Mean over {len(TEST_SEEDS)} fresh worlds (%) ===")
    for b in ["fixed rule", "5% review budget", "7% review budget", "same alert volume as P2 (about 7%)"]: print(summ.loc[[b]].drop(columns="roc_auc").to_string(), "\n")
    for budget in ["5% review budget", "7% review budget"]:
        g = lambda c: res[(res.condition == c) & (res.budget == budget)].set_index("seed")
        for m in ["detection_%", "F1_%", "false_pos_%"]:
            dd = g("E2 + device/location")[m] - g("E0 main model")[m]; half = stats.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
            print(f"   device/location minus main model, {budget:<17} {m:<12} {dd.mean():+6.2f} [95% CI {dd.mean() - half:+.2f}, {dd.mean() + half:+.2f}]")
    fa = []
    for d in tests:
        pol = policies(d); vol = float(pol["P2 + device/location"].mean()); legit = (d["is_anomaly"] == 0).to_numpy(); dev = ((d["device_known"] == 0) | (d["location_trusted"] == 0)).to_numpy(); un = (d["justified_access"] == 0).to_numpy()
        for name, flag in [("P2 + device/location (rule)", pol["P2 + device/location"]), ("E2 + device/location (model, same volume)", topk(-models["E2 + device/location"].decision_function(d[FEATS["E2 + device/location"]]), vol))]:
            f = flag & legit; fa.append({"condition": name, "false alarms": f.sum(), "unjustified access (genuine emergencies)": (f & un & ~dev).sum(), "unknown device or location (legitimate noise)": (f & dev).sum(), "other": (f & ~un & ~dev).sum()})
    fa = pd.DataFrame(fa).groupby("condition", sort=False).sum(); fa.to_csv(OUT / "care_device_false_alarms.csv"); print("\n=== Where the false alarms come from (summed over 20 worlds) ===\n" + fa.to_string())
    sens = []
    for rate in SENS_RATES:
        _, mm = train(staff, patients, rate); rs = []
        for s in SENS_SEEDS: d, _, _ = make(staff, patients, 30, s, rate); rs += [{"seed": s, **r} for r in evaluate(d, mm) if r["budget"] in ("7% review budget", "fixed rule")]
        r = pd.DataFrame(rs).groupby("condition")[["detection_%", "stolen_credentials_%", "false_pos_%", "F1_%"]].mean()
        for cond in ["P0 Basic policy (main)", "P2 + device/location", "E0 main model", "E2 + device/location"]: sens.append({"unregistered_device_rate": rate, "condition": cond, **r.loc[cond].to_dict()})
    sens = pd.DataFrame(sens); sens.round(3).to_csv(OUT / "care_device_sensitivity.csv", index=False)
    print("\n=== Sensitivity: F1 (%) as legitimate device noise changes (10 worlds each; models at a 7% review budget) ===\n" + sens.pivot_table(index="unregistered_device_rate", columns="condition", values="F1_%").round(1)[["P0 Basic policy (main)", "P2 + device/location", "E0 main model", "E2 + device/location"]].to_string())

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8)); lab = "7% review budget"; conds = ["P0 Basic policy (main)", "P2 + device/location", "E0 main model", "E2 + device/location"]
    short = ["Basic\npolicy", "Basic\n+device", "Model\nmain", "Model\n+device"]; col = ["#999999", "#4C72B0", "#F0B27A", "#D55E00"]
    get = lambda c, m: (summ.loc[("fixed rule", c), m] if c.startswith("P") else summ.loc[(lab, c), m])
    for j, (m, al) in enumerate([("off_hours_%", .45), ("cross_ward_%", .75), ("stolen_credentials_%", 1.0)]): ax[0].bar(np.arange(4) + (j - 1) * .26, [get(c, m) for c in conds], .26, color=col, alpha=al, edgecolor="black", linewidth=.4)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(short); ax[0].set_ylim(0, 128); ax[0].set_title("Detection by scenario (%)", fontsize=11)
    ax[0].legend(handles=[Patch(facecolor="#777", alpha=a, label=l) for a, l in [(.45, "Off-hours"), (.75, "Cross-ward"), (1.0, "Stolen credentials (extension)")]], loc="upper center", ncol=3, fontsize=7.5)
    for j, (m, al) in enumerate([("F1_%", 1.0), ("false_pos_%", .55)]): ax[1].bar(np.arange(4) + (j - .5) * .38, [get(c, m) for c in conds], .38, color=col, alpha=al, edgecolor="black", linewidth=.4)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(short); ax[1].set_ylim(0, 128); ax[1].set_title("F1 and false positives (%)", fontsize=11)
    ax[1].legend(handles=[Patch(facecolor="#777", label="F1"), Patch(facecolor="#777", alpha=.55, label="False positives")], loc="upper center", ncol=2, fontsize=8.5)
    for cond, c, mk in [("E2 + device/location", "#D55E00", "o"), ("E0 main model", "#F0B27A", "s"), ("P2 + device/location", "#4C72B0", "^"), ("P0 Basic policy (main)", "#999999", "d")]:
        g = sens[sens.condition == cond]; ax[2].plot(100 * g["unregistered_device_rate"], g["F1_%"], mk + "-", color=c, label=cond)
    ax[2].set_xlabel("Legitimate events on an unregistered device (%)"); ax[2].set_ylabel("F1 (%)"); ax[2].set_ylim(0, 105); ax[2].set_title("F1 against legitimate device noise\n(models at a 7% review budget)", fontsize=10.5); ax[2].legend(fontsize=8)
    fig.suptitle("Device and location on top of the main design (extension; 20 fresh worlds; models at a fixed 7% review budget)", fontsize=11); fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_care_device.png", dpi=300); fig.savefig(OUT / "fig_care_device.pdf"); print(f"\nSaved to {OUT}/")
