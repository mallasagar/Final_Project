"""
compare_datasets.py
STEP 3 of 3 - COMPARE the results of every dataset in testing/datasets/, side by side.
Writes to testing/comparison/:
    dataset_comparison.csv        one row per dataset, the key results of all three conditions
    dataset_comparison_long.csv   every metric, every condition, every dataset
    profile_summary.csv           averages per profile (when you used --profiles)
    comparison_report.txt         the highlights in plain words
    figures/compare_detection.png, figures/compare_heatmap.png

    py src/testing/compare_datasets.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as C

ORDER = ["Traditional", "Basic ZTA", "Enhanced ZTA"]
COL = {"Traditional": "#999999", "Basic ZTA": "#0072B2", "Enhanced ZTA": "#D55E00"}


def load():
    parts = []
    for f in sorted(d for d in C.DATASETS.iterdir() if d.is_dir()):
        if not (f / "results.csv").exists():
            print(f"  skipping {f.name}: not tested yet"); continue
        r = pd.read_csv(f / "results.csv"); cfg = json.load(open(f / "config.json"))
        r["seed"], r["days"], r["anomaly_rate_set"] = cfg["seed"], cfg["days"], cfg["anomaly_rate"]
        parts.append(r)
    if not parts:
        raise SystemExit("No tested datasets. Run test_datasets.py first.")
    return pd.concat(parts, ignore_index=True)


def wide(long):
    rows = []
    for name, g in long.groupby("dataset", sort=True):
        g = g.set_index("condition"); e, b, t = g.loc["Enhanced ZTA"], g.loc["Basic ZTA"], g.loc["Traditional"]
        rows.append({"dataset": name, "profile": e["profile"], "seed": int(e["seed"]), "days": int(e["days"]),
                     "events": int(e["events"]), "anomalies": int(e["anomalies"]), "anomaly_rate_%": round(100 * e["anomalies"] / e["events"], 2),
                     "detection_Traditional": t.detection_overall, "detection_Basic": b.detection_overall, "detection_Enhanced": e.detection_overall,
                     "gap_Enhanced_minus_Basic": e.detection_overall - b.detection_overall,
                     "Enh_off_hours": e.detection_off_hours, "Enh_cross_ward": e.detection_cross_ward, "Enh_false_pos": e.false_positive,
                     "Enh_precision": e.precision, "Enh_F1": e.f1, "Basic_F1": b.f1, "Enh_accuracy": e.accuracy, "Enh_roc_auc": e.roc_auc,
                     "records_Traditional": t.records_incl_missed, "records_Basic": b.records_incl_missed, "records_Enhanced": e.records_incl_missed})
    out = pd.DataFrame(rows); auc = out["Enh_roc_auc"].copy()
    out = out.round(2); out["Enh_roc_auc"] = auc.round(4)            # keep AUC to four decimals
    return out


def fig_detection(w, path):
    x = np.arange(len(w)); fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(w) + 3), 4.8))
    for c, col in [("Traditional", "detection_Traditional"), ("Basic ZTA", "detection_Basic"), ("Enhanced ZTA", "detection_Enhanced")]:
        ax.plot(x, w[col], "o-", color=COL[c], label=c, lw=1.6, ms=6)
    labels = [n.split("_", 1)[1] if len(w) <= 24 else str(i + 1) for i, n in enumerate(w.dataset)]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40 if len(w) <= 24 else 0, ha="right" if len(w) <= 24 else "center", fontsize=8.5)
    ax.set_ylabel("Overall detection rate (%)"); ax.set_ylim(0, 100); ax.legend(loc="center right")
    ax.set_title("Detection rate on each dataset"); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_heatmap(w, path):
    cols = {"Enh_roc_auc": "ROC-AUC", "detection_Enhanced": "Detection\noverall %", "Enh_cross_ward": "Detection\ncross-ward %", "Enh_off_hours": "Detection\noff-hours %",
            "Enh_false_pos": "False\npositives %", "Enh_F1": "F1 %", "records_Enhanced": "Records exposed\n(incl. missed)"}
    v = w.set_index("dataset")[list(cols)].copy(); v["Enh_roc_auc"] = v["Enh_roc_auc"] * (100 if v["Enh_roc_auc"].max() <= 1.5 else 1)
    norm = (v - v.min()) / (v.max() - v.min()).replace(0, 1)
    fig, ax = plt.subplots(figsize=(10.5, max(3.5, 0.42 * len(v) + 1.8)))
    im = ax.imshow(norm.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, f"{v.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8.5, color="white" if norm.iloc[i, j] > .6 else "black")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(list(cols.values()), fontsize=9)
    ax.set_yticks(range(len(v))); ax.set_yticklabels(v.index, fontsize=8.5); ax.xaxis.tick_top()
    ax.set_title("Enhanced ZTA on each dataset (darker = higher value within the column)", pad=34, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def main():
    C.COMPARISON.mkdir(parents=True, exist_ok=True); (C.COMPARISON / "figures").mkdir(exist_ok=True)
    long = load(); w = wide(long)
    long.round(3).to_csv(C.COMPARISON / "dataset_comparison_long.csv", index=False); w.to_csv(C.COMPARISON / "dataset_comparison.csv", index=False)
    fig_detection(w, C.COMPARISON / "figures" / "compare_detection.png"); fig_heatmap(w, C.COMPARISON / "figures" / "compare_heatmap.png")

    show = w[["dataset", "anomalies", "detection_Traditional", "detection_Basic", "detection_Enhanced", "gap_Enhanced_minus_Basic", "Enh_cross_ward", "Enh_false_pos", "Enh_F1"]]
    show = show.rename(columns={"detection_Traditional": "Trad %", "detection_Basic": "Basic %", "detection_Enhanced": "Enh %", "gap_Enhanced_minus_Basic": "Enh-Basic",
                                "Enh_cross_ward": "Enh x-ward %", "Enh_false_pos": "Enh FP %", "Enh_F1": "Enh F1"})
    print("\n" + "=" * 110 + "\nEACH DATASET, SIDE BY SIDE  (detection rates are % of that dataset's anomalies)\n" + "=" * 110)
    print(show.to_string(index=False))

    lines = [f"COMPARISON OF {len(w)} DATASET(S)", ""]
    wins = int((w.gap_Enhanced_minus_Basic > 0).sum())
    lines.append(f"Enhanced ZTA detected more than Basic ZTA in {wins} of {len(w)} datasets.")
    be, we = w.loc[w.detection_Enhanced.idxmax()], w.loc[w.detection_Enhanced.idxmin()]
    lines.append(f"Enhanced ZTA's detection ranged from {we.detection_Enhanced:.1f}% ({we.dataset}) to {be.detection_Enhanced:.1f}% ({be.dataset}).")
    lines.append(f"Its false-positive rate ranged from {w.Enh_false_pos.min():.2f}% to {w.Enh_false_pos.max():.2f}% (Basic ZTA and Traditional: 0% in every dataset where the rule has no false alarms).")
    lines.append(f"Its cross-ward detection ranged from {w.Enh_cross_ward.min():.1f}% to {w.Enh_cross_ward.max():.1f}%.")
    lines.append(f"Records exposed (incl. missed), Enhanced ZTA: {w.records_Enhanced.min():.0f} to {w.records_Enhanced.max():.0f}; Basic ZTA: {w.records_Basic.min():.0f} to {w.records_Basic.max():.0f}.")
    if w.profile.nunique() > 1:
        ps = w.groupby("profile", sort=False).agg(datasets=("dataset", "count"), anomalies=("anomalies", "mean"), Trad=("detection_Traditional", "mean"),
                                                  Basic=("detection_Basic", "mean"), Enhanced=("detection_Enhanced", "mean"), gap=("gap_Enhanced_minus_Basic", "mean"),
                                                  Enh_cross_ward=("Enh_cross_ward", "mean"), Enh_FP=("Enh_false_pos", "mean"), Enh_F1=("Enh_F1", "mean")).round(1)
        ps.to_csv(C.COMPARISON / "profile_summary.csv"); lines += ["", "By profile (mean over its datasets):", ps.to_string()]
        print("\n" + "=" * 110 + "\nBY PROFILE\n" + "=" * 110); print(ps.to_string())
        worst = ps.gap.idxmin(); lines.append(f"\nThe smallest Enhanced-over-Basic gap is in the '{worst}' profile ({ps.gap.min():.1f} points).")
    if (w.days.min() < 30) or (w.anomalies.min() < 150):
        lines.append("\nNote: datasets with few anomalies give noisy rates; compare them with care.")
    (C.COMPARISON / "comparison_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[:6]))
    print(f"\nSaved to {C.COMPARISON}:\n  dataset_comparison.csv, dataset_comparison_long.csv, comparison_report.txt, figures/compare_detection.png, figures/compare_heatmap.png")


if __name__ == "__main__":
    main()
