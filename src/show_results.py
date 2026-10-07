from pathlib import Path

import pandas as pd

R = Path("results")
ORDER = ["Traditional", "Basic ZTA", "Enhanced ZTA"]


def load(name):
    p = R / name
    if not p.exists():
        print(f"  (missing: {p} - run the pipeline first)\n")
        return None
    return pd.read_csv(p)


def human(seconds):
    if seconds >= 86400: return f"{seconds / 86400:.1f} days"
    if seconds >= 3600: return f"{seconds / 3600:.2f} hours"
    return f"{seconds:.2f} seconds"


def mean_sd(series):
    return f"{series.mean():5.1f} +/- {series.std(ddof=1):4.1f}"


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


res = load("experiment_results_independent.csv")
section("1. DETECTION AND FALSE ALARMS  (20 independent 30-day datasets; mean +/- spread across datasets)")
if res is not None:
    rows = []
    for c in ORDER:
        d = res[res.condition == c]
        rows.append({"condition": c,
                     "off-hours detection %": mean_sd(d[d.scenario == "off_hours"].detection_rate),
                     "cross-ward detection %": mean_sd(d[d.scenario == "cross_ward"].detection_rate),
                     "false positives %": mean_sd(d.groupby("trial").false_positive_rate.mean())})
    print(pd.DataFrame(rows).to_string(index=False))

cls = load("classification_metrics_independent.csv")
section("2. OVERALL CLASSIFICATION  (all events, both scenarios; mean over the 20 datasets)")
if cls is not None:
    t = cls.groupby("condition")[["accuracy_%", "precision_%", "recall_%", "F1_%", "false_positive_%"]].mean().round(1).reindex(ORDER)
    print(t.to_string())
    print(f"\n  Accuracy of a system that flags nothing: {cls['all_normal_baseline_%'].mean():.1f}%  <- accuracy alone says little")

section("3. SPEED AND EXPOSURE  (Traditional and Basic ZTA timings are ASSUMED ranges; Enhanced ZTA's are modelled latencies)")
if res is not None:
    m = res.groupby("condition")[["MTTD", "MTTR", "records_exposed", "records_exposed_incl_missed"]].mean().reindex(ORDER)
    out = pd.DataFrame({"MTTD": m.MTTD.map(human), "MTTR": m.MTTR.map(human),
                        "records exposed (detected)": m.records_exposed.round(1),
                        "records exposed (incl. missed)": m.records_exposed_incl_missed.round(1)})
    print(out.to_string())

hold = load("holdout_summary.csv")
section("4. DOES THE MODEL HOLD UP ON DATA IT NEVER SAW?  (ROC-AUC: 1.0 = perfect, 0.5 = guessing)")
if hold is not None:
    tr, ho = hold[hold.seed.astype(str) == "training"], hold[hold.seed.astype(str) != "training"]
    print(f"  Training data:          ROC-AUC {tr.roc_auc.iloc[0]:.3f}   detection {tr['detection_rate_%'].iloc[0]:.1f}%   false positives {tr['false_positive_%'].iloc[0]:.1f}%")
    print(f"  {len(ho)} held-out datasets:    ROC-AUC {ho.roc_auc.mean():.3f} +/- {ho.roc_auc.std(ddof=1):.3f}   "
          f"detection {ho['detection_rate_%'].mean():.1f}% +/- {ho['detection_rate_%'].std(ddof=1):.1f}   "
          f"false positives {ho['false_positive_%'].mean():.1f}%")

stats = load("statistical_summary_experiment_results_independent.csv")
section("5. STATISTICAL TESTS  (Kruskal-Wallis across the three conditions; see the caveat on timing metrics)")
if stats is not None:
    s = stats[["metric", "H", "p_value", "epsilon_squared", "effect_size"]].copy()
    s["p_value"] = s.p_value.map(lambda p: "< .001" if p < .001 else f"{p:.3f}")
    print(s.round(3).to_string(index=False))
    print("\n  Caveat: MTTD, MTTR and records exposed differ by construction for Traditional and Basic ZTA, because")
    print("  their timings are assumed ranges. Only the detection and false-positive tests are empirical.")
print()
