"""
run_roster_extension.py   (EXTENSION - writes only to extension_roster/, never to data/ or results/)
Does a staff roster improve detection? Same hospital, same two scenarios, same Isolation Forest settings; only the roster
information differs.

Conditions compared on 20 independent 30-day datasets (seeds 60000-60019), each with its own roster:
  Traditional                   hour window, applied by occasional manual audit (30% catch)
  Basic ZTA                     hour window, applied continuously
  Basic ZTA + ward policy       window OR any access outside the user's home ward
  Basic ZTA + roster policy     window OR any access outside the home ward and the rostered ward
  Enhanced, no roster           Isolation Forest, features: hours beyond shift, ward match, hour sin/cos
  Enhanced + roster             Isolation Forest, ward match replaced by 'authorised ward' (roster lookup)
Both models: 200 trees, contamination 0.05 (a fixed 5% review budget, no label-based tuning), trained once on a 90-day world (seed 42).
A sensitivity analysis varies the share of legitimate cross-ward access that is NOT on the roster (ad-hoc/emergency).

Run from the project root:   py src/extension/run_roster_extension.py
Outputs (extension_roster/): roster_extension_results.csv, roster_extension_summary.csv, roster_extension_sensitivity.csv,
                             roster_extension_budget.csv, training_access_log.csv, training_roster.csv, fig_roster_extension.png / .pdf
"""
import os
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
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from roster_world import generate_world, engineer_roster_features, ROSTER_FEATURES, NO_ROSTER_FEATURES, ADHOC_RATE

OUT = Path("extension_roster")
TEST_SEEDS = list(range(60000, 60020))
SENS_RATES = [0.005, 0.015, 0.03, 0.06]
SENS_SEEDS = TEST_SEEDS[:10]
ORDER = ["Traditional", "Basic ZTA", "Basic ZTA + ward policy", "Basic ZTA + roster policy", "Enhanced, no roster", "Enhanced + roster"]


def make(patients, staff, days, seed, adhoc=ADHOC_RATE):
    logs, roster = generate_world(patients, staff, days, seed, adhoc_rate=adhoc)
    return engineer_roster_features(logs, staff, roster), logs, roster


def checks(d, label):
    """Data checks: the extension must not leak labels into the roster information."""
    legit_mismatch = d[(d.is_anomaly == 0) & (d.ward_match == 0)]
    cross = d[d.anomaly_type == "cross_ward"]
    ok = [("anomaly rate close to 5%", 0.04 <= d.is_anomaly.mean() <= 0.06),
          ("every cross-ward anomaly is on an unrostered ward (authorised_ward = 0)", bool((cross.authorised_ward == 0).all())),
          ("every off-hours anomaly is on the user's own ward", bool((d[d.anomaly_type == "off_hours"].ward_match == 1).all())),
          ("rostered cover exists (legitimate mismatch events that ARE authorised)", bool((legit_mismatch.authorised_ward == 1).any())),
          ("ad-hoc legitimate access exists (legitimate mismatch events that are NOT authorised)", bool((legit_mismatch.authorised_ward == 0).any())),
          ("no label column is a model feature", not ({"is_anomaly", "anomaly_type"} & set(ROSTER_FEATURES + NO_ROSTER_FEATURES)))]
    print(f"  Data checks on {label}:")
    for name, passed in ok: print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
    if not all(p for _, p in ok): raise SystemExit("A data check failed. Stopping.")
    share = 100 * (legit_mismatch.authorised_ward == 1).mean()
    print(f"    legitimate cross-ward events: {len(legit_mismatch)}; on the roster: {share:.1f}%; ad-hoc (not rostered): {100 - share:.1f}%; cross-ward anomalies: {len(cross)}")
    return share


def metrics(d, flag):
    y, ct = d["is_anomaly"].to_numpy(), d["anomaly_type"].to_numpy(); tp, fp = (flag & (y == 1)).sum(), (flag & (y == 0)).sum()
    p = tp / max(tp + fp, 1); r = tp / y.sum()
    return {"detection_%": 100 * r, "off_hours_%": 100 * flag[ct == "off_hours"].mean(), "cross_ward_%": 100 * flag[ct == "cross_ward"].mean(),
            "false_pos_%": 100 * fp / (y == 0).sum(), "precision_%": 100 * p, "F1_%": 100 * 2 * p * r / max(p + r, 1e-9)}


def evaluate(d, seed, m0, m1):
    window = ((d["hour"] < 7) | (d["hour"] > 20)).to_numpy()
    rng = np.random.default_rng(seed + 7)
    rows = {"Traditional": metrics(d, window & (rng.random(len(d)) < 0.30)), "Basic ZTA": metrics(d, window),
            "Basic ZTA + ward policy": metrics(d, window | (d["ward_match"] == 0).to_numpy()),
            "Basic ZTA + roster policy": metrics(d, window | (d["authorised_ward"] == 0).to_numpy())}
    y = d["is_anomaly"].to_numpy(); legit_x = ((d["is_anomaly"] == 0) & (d["ward_match"] == 0)).to_numpy(); cw = (d["anomaly_type"] == "cross_ward").to_numpy()
    for name, m, cols in [("Enhanced, no roster", m0, NO_ROSTER_FEATURES), ("Enhanced + roster", m1, ROSTER_FEATURES)]:
        s = -m.decision_function(d[cols]); r = metrics(d, m.predict(d[cols]) == -1)
        sel = cw | legit_x                                                                    # cross-ward anomalies vs legitimate cross-ward access
        r.update({"roc_auc": roc_auc_score(y, s), "avg_precision": average_precision_score(y, s), "anomaly_vs_legit_cross_ward_auc": roc_auc_score(cw[sel], s[sel])})
        rows[name] = r
    return rows


def train(patients, staff, adhoc=ADHOC_RATE):
    tr, logs, roster = make(patients, staff, 90, 42, adhoc)
    fit = lambda cols: IsolationForest(n_estimators=200, contamination=0.05, random_state=42).fit(tr[cols])
    return tr, fit(NO_ROSTER_FEATURES), fit(ROSTER_FEATURES), logs, roster


if __name__ == "__main__":
    staff, patients = pd.read_csv("data/staff.csv"), pd.read_csv("data/patients_with_wards.csv")
    OUT.mkdir(exist_ok=True)
    print("=" * 78 + "\nROSTER EXTENSION  (separate from the main experiment; writes only to extension_roster/)\n" + "=" * 78)
    tr, m0, m1, logs, roster = train(patients, staff); logs.to_csv(OUT / "training_access_log.csv", index=False); roster.to_csv(OUT / "training_roster.csv", index=False)
    checks(tr, "the 90-day training world")
    print(f"\nTesting on {len(TEST_SEEDS)} independent 30-day worlds (seeds {TEST_SEEDS[0]}-{TEST_SEEDS[-1]}), each with its own roster ...")
    rows, tests = [], []
    for s in TEST_SEEDS:
        d, _, _ = make(patients, staff, 30, s); tests.append((d, s))
        for cond, r in evaluate(d, s, m0, m1).items(): rows.append({"seed": s, "condition": cond, **r})
    res = pd.DataFrame(rows); res.round(4).to_csv(OUT / "roster_extension_results.csv", index=False)
    metric_cols = ["detection_%", "off_hours_%", "cross_ward_%", "false_pos_%", "precision_%", "F1_%", "roc_auc", "avg_precision", "anomaly_vs_legit_cross_ward_auc"]
    summ = []
    for c in ORDER:
        g = res[res.condition == c]; row = {"condition": c}
        for m in metric_cols:
            if m in g and g[m].notna().any(): v = g[m].dropna(); row[m] = v.mean(); row[m + "_ci95"] = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        summ.append(row)
    summ = pd.DataFrame(summ); summ.round(4).to_csv(OUT / "roster_extension_summary.csv", index=False)
    show = summ[["condition", "detection_%", "off_hours_%", "cross_ward_%", "false_pos_%", "precision_%", "F1_%"]].round(1)
    print("\n=== Mean over 20 independent datasets (detection, false positives, precision, F1 in %) ===\n" + show.to_string(index=False))
    mod = summ[summ.condition.str.startswith("Enhanced")][["condition", "roc_auc", "avg_precision", "anomaly_vs_legit_cross_ward_auc"]].round(3)
    print("\n=== Ranking quality of the two models (threshold-free) ===\n" + mod.to_string(index=False))
    print("   'anomaly_vs_legit_cross_ward_auc': can the model tell a cross-ward ANOMALY from legitimate cross-ward access? (0.5 = no)")
    w = res[res.condition == "Enhanced + roster"].set_index("seed"); wo = res[res.condition == "Enhanced, no roster"].set_index("seed")
    for m in ["detection_%", "cross_ward_%", "false_pos_%", "F1_%"]:
        d = (w[m] - wo[m]); half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        print(f"   roster minus no roster, {m:<13} {d.mean():+6.2f}  [95% CI {d.mean() - half:+.2f}, {d.mean() + half:+.2f}]   roster better in {int((d > 0).sum() if m != 'false_pos_%' else (d < 0).sum())}/{len(d)} datasets")

    # ---- fair comparison without tuning on labels: fixed review budgets, and the SAME alert volume as the roster rule ----
    def topk(score, k):
        flag = np.zeros(len(score), bool); flag[np.argsort(-score, kind="stable")[:k]] = True; return flag
    budgets = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]; brow = []
    for name, m, cols in [("Enhanced, no roster", m0, NO_ROSTER_FEATURES), ("Enhanced + roster", m1, ROSTER_FEATURES)]:
        for b in budgets:
            ms = [metrics(d, topk(-m.decision_function(d[cols]), int(round(b * len(d))))) for d, _ in tests]
            brow.append({"model": name, "review_budget_%": 100 * b, **pd.DataFrame(ms).mean().to_dict()})
        ms, vol = [], []
        for d, _ in tests:
            rule = ((d["hour"] < 7) | (d["hour"] > 20)).to_numpy() | (d["authorised_ward"] == 0).to_numpy()
            ms.append(metrics(d, topk(-m.decision_function(d[cols]), int(rule.sum())))); vol.append(100 * rule.mean())
        brow.append({"model": name, "review_budget_%": float(np.mean(vol)), "note": "same alert volume as the roster rule", **pd.DataFrame(ms).mean().to_dict()})
    brow = pd.DataFrame(brow); brow.round(4).to_csv(OUT / "roster_extension_budget.csv", index=False)
    print("\n=== Fixed review budgets (top x% of events by score; no labels used to choose the cut-off) ===")
    print(brow[brow.note.isna()][["model", "review_budget_%", "detection_%", "cross_ward_%", "false_pos_%", "precision_%", "F1_%"]].round(1).to_string(index=False))
    print("\n=== Same number of alerts as the roster rule (about %.1f%% of events) ===" % brow[brow.note.notna()]["review_budget_%"].mean())
    print(brow[brow.note.notna()][["model", "detection_%", "cross_ward_%", "false_pos_%", "precision_%", "F1_%"]].round(1).to_string(index=False))
    print("   (the roster rule itself: detection %.1f%%, false positives %.1f%%, F1 %.1f%%)" % tuple(summ.loc[summ.condition == "Basic ZTA + roster policy", c].iloc[0] for c in ["detection_%", "false_pos_%", "F1_%"]))

    print("\n=== Sensitivity: how much legitimate cross-ward access is NOT on the roster? ===")
    sens = []
    for rate in SENS_RATES:
        trs, a0, a1, _, _ = train(patients, staff, rate)
        results = []
        for s in SENS_SEEDS:
            d, _, _ = make(patients, staff, 30, s, rate); results.append(evaluate(d, s, a0, a1))
        lm = trs[(trs.is_anomaly == 0) & (trs.ward_match == 0)]; on_roster = 100 * (lm.authorised_ward == 1).mean()
        for cond in ["Basic ZTA + ward policy", "Basic ZTA + roster policy", "Enhanced, no roster", "Enhanced + roster"]:
            v = pd.DataFrame([r[cond] for r in results]).mean(); sens.append({"adhoc_rate": rate, "legit_cross_ward_on_roster_%": on_roster, "condition": cond, **v.to_dict()})
    sens = pd.DataFrame(sens); sens.round(4).to_csv(OUT / "roster_extension_sensitivity.csv", index=False)
    piv = sens.pivot_table(index=["adhoc_rate", "legit_cross_ward_on_roster_%"], columns="condition", values="F1_%").round(1)[["Basic ZTA + ward policy", "Basic ZTA + roster policy", "Enhanced, no roster", "Enhanced + roster"]]
    print("F1 (%) by condition, 10 datasets per row:\n" + piv.to_string())

    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8)); x = np.arange(len(ORDER)); colour = ["#BBBBBB", "#999999", "#777777", "#4C72B0", "#F0B27A", "#D55E00"]
    for a, cols, title in [(ax[0], [("detection_%", "Overall detection"), ("cross_ward_%", "Cross-ward detection")], "Detection (%)"), (ax[1], [("F1_%", "F1"), ("precision_%", "Precision")], "F1 and precision (%)")]:
        for j, (c, lab) in enumerate(cols): a.bar(x + (j - .5) * .38, [summ.loc[summ.condition == k, c].iloc[0] for k in ORDER], .38, color=colour, alpha=1 if j == 0 else .55, edgecolor="black", linewidth=.4, label=lab)
        a.set_xticks(x); a.set_xticklabels(["Trad.", "Basic", "Basic\n+ward", "Basic\n+roster", "Enh.\nno roster", "Enh.\n+roster"], fontsize=8.5); a.set_ylim(0, 128); a.set_title(title, fontsize=11)
        from matplotlib.patches import Patch
        a.legend(handles=[Patch(facecolor="#777", label=cols[0][1]), Patch(facecolor="#777", alpha=.5, label=cols[1][1])], loc="upper center", ncol=2, fontsize=8.5)
    for cond, c, mk in [("Enhanced + roster", "#D55E00", "o"), ("Enhanced, no roster", "#F0B27A", "s"), ("Basic ZTA + roster policy", "#4C72B0", "^"), ("Basic ZTA + ward policy", "#777777", "d")]:
        g = sens[sens.condition == cond]; ax[2].plot(g["legit_cross_ward_on_roster_%"], g["F1_%"], mk + "-", color=c, label=cond)
    ax[2].set_xlabel("Legitimate cross-ward access that is on the roster (%)"); ax[2].set_ylabel("F1 (%)"); ax[2].set_title("F1 against roster coverage", fontsize=11); ax[2].legend(fontsize=8); ax[2].set_ylim(0, 105)
    for model, c, mk in [("Enhanced + roster", "#D55E00", "o"), ("Enhanced, no roster", "#F0B27A", "s")]:
        g = brow[(brow.model == model) & brow.note.isna()]; ax[3].plot(g["review_budget_%"], g["F1_%"], mk + "-", color=c, label=model)
    for cond, c, ls in [("Basic ZTA + roster policy", "#4C72B0", "--"), ("Basic ZTA + ward policy", "#777777", ":")]:
        ax[3].axhline(summ.loc[summ.condition == cond, "F1_%"].iloc[0], color=c, ls=ls, label=cond + " (a fixed rule)")
    ax[3].set_xlabel("Review budget (% of events the model may flag)"); ax[3].set_ylabel("F1 (%)"); ax[3].set_ylim(0, 105); ax[3].set_title("The model at different review budgets", fontsize=11); ax[3].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Roster extension (20 independent datasets; panels A-C use a fixed 5% review budget)", fontsize=11); fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_roster_extension.png", dpi=300); fig.savefig(OUT / "fig_roster_extension.pdf")
    print(f"\nSaved to {OUT}/: roster_extension_results.csv, roster_extension_summary.csv, roster_extension_sensitivity.csv, roster_extension_budget.csv, training_access_log.csv, training_roster.csv, fig_roster_extension.png")
