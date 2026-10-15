"""
feature_search.py
Is there ANY improvement left from combining features? An exhaustive-as-possible, disciplined search.

Pool: 17 base columns that carry no label (hour of access, authorised/rostered ward flags, role, action, cover day, access count, ...).
Candidates: every base column, plus EVERY pairwise product of base columns (a product of two flags is an AND; of a flag and a number, a scaling).
Search: starting from the four main features, add the single candidate that most improves F1 (at a fixed 5% review budget) on VALIDATION
worlds (seeds 70000-70004), three times (greedy forward selection). Nothing is tuned on the test worlds.
Report: the chosen sets on 20 fresh TEST worlds (seeds 80000-80019), next to the main model, plus
  - 'all features at once' (every base column and every pairwise product),
  - a supervised second stage that gets the model's own SCORE as an input (uses labels: a ceiling, not a deployable model),
  - fusing the model's flags with Basic ZTA's policy flags (union and intersection).
Every candidate's validation score is saved, and the winner's curse (validation score vs test score) is shown.
Every subset of 136 candidates cannot be tried (more than 2^130 subsets); single and pairwise combinations are all covered.

Run from the project root, after anomaly_model.py:   py src/feature_search.py     (about 5-8 minutes)
Saves results/feature_search_*.csv
"""
import itertools
import os
import sys
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
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import access_log_generator as gen
from feature_engineering import engineer_features, FEATURE_COLS
from run_experiment import basic_policy_flag

BASE = ["hours_beyond_shift", "clock_hour", "hour_sin", "hour_cos", "access_count_1h", "authorised_ward", "unauthorised_ward", "ward_match",
        "rostered_match", "cover_day", "care_relationship", "outside_shift", "is_doctor", "is_nurse", "is_admin", "is_export", "is_write"]
VAL_SEEDS, TEST_SEEDS = list(range(70000, 70005)), list(range(80000, 80020))
BUDGET, STEPS = 0.05, 3


def prep(d):
    """Add the base columns (flags, roles, actions) and every pairwise product of them to a featured log."""
    d = d.copy()
    d["clock_hour"] = d["hour"]; d["rostered_match"] = (d["patient_ward"] == d["rostered_ward"]).astype(int)
    d["cover_day"] = (d["rostered_ward"] != d["assigned_ward"]).astype(int); d["unauthorised_ward"] = 1 - d["authorised_ward"]
    d["outside_shift"] = (d["hours_beyond_shift"] >= 1).astype(int)
    for r in ("Doctor", "Nurse", "Admin"): d["is_" + r.lower()] = (d["role"] == r).astype(int)
    d["is_export"], d["is_write"] = (d["action"] == "export").astype(int), (d["action"] == "write").astype(int)
    pairs = {f"{a} x {b}": d[a].to_numpy() * d[b].to_numpy() for a, b in itertools.combinations(BASE, 2)}
    return pd.concat([d, pd.DataFrame(pairs, index=d.index)], axis=1)


def world(staff, patients, seed, days):
    """Generate one dataset and prepare all candidate features."""
    return prep(engineer_features(gen.generate_access_logs(patients, staff, n_days=days, events_per_day=gen.EVENTS_PER_DAY, seed=seed), staff)[0])


def topk(score, frac):
    """Flag the top fraction of events by score (a review budget fixed in advance)."""
    k = int(round(frac * len(score))); f = np.zeros(len(score), bool); f[np.argsort(-score, kind="stable")[:k]] = True; return f


def metr(d, flag):
    """Detection, cross-ward detection, false positives, precision and F1 for one set of flags."""
    y, ct = d["is_anomaly"].to_numpy(), d["anomaly_type"].to_numpy(); tp, fp = (flag & (y == 1)).sum(), (flag & (y == 0)).sum()
    p = tp / max(tp + fp, 1); r = tp / y.sum()
    return {"detection_%": 100 * r, "cross_ward_%": 100 * flag[ct == "cross_ward"].mean(), "false_pos_%": 100 * fp / (y == 0).sum(), "precision_%": 100 * p, "F1_%": 100 * 2 * p * r / max(p + r, 1e-9)}


def fit(train, cols, trees):
    """Fit an Isolation Forest on the chosen columns."""
    return IsolationForest(n_estimators=trees, contamination=0.05, random_state=42).fit(train[cols])


def val_f1(model, cols, val):
    """Mean F1 of a model over the validation datasets, at the fixed review budget: the score the greedy search maximises."""
    return float(np.mean([metr(d, topk(-model.decision_function(d[cols]), BUDGET))["F1_%"] for d in val]))


if __name__ == "__main__":
    staff, patients = pd.read_csv("data/staff.csv"), pd.read_csv("data/patients_with_wards.csv")
    os.makedirs("results", exist_ok=True)
    train = prep(engineer_features(gen.generate_access_logs(patients, staff, n_days=90, events_per_day=gen.EVENTS_PER_DAY, seed=42), staff)[0])
    val = [world(staff, patients, s, 30) for s in VAL_SEEDS]; test = [world(staff, patients, s, 30) for s in TEST_SEEDS]
    # Every base column and every pairwise product, except constant columns.
    pool = [c for c in train.columns if (" x " in c or c in BASE) and train[c].std() > 0]
    candidates = [c for c in pool if c not in FEATURE_COLS]
    print(f"Pool: {len(BASE)} base columns -> {len(candidates)} candidate features (every base column and every pairwise product).\n")

    # Baseline: the main four features, scored on the validation datasets.
    base_val = val_f1(fit(train, FEATURE_COLS, 100), FEATURE_COLS, val); print(f"Validation F1 of the main four features (100 trees): {base_val:.2f}%")
    chosen, history, step1 = list(FEATURE_COLS), [], None
    # Greedy forward selection on VALIDATION data only: at each step add the one candidate that raises F1 most.
    for step in range(1, STEPS + 1):
        scores = {}
        for c in candidates:
            if c in chosen: continue
            cols = chosen + [c]; scores[c] = val_f1(fit(train, cols, 100), cols, val)
        best = max(scores, key=scores.get)
        if step == 1: step1 = pd.Series(scores).sort_values(ascending=False)
        history.append((step, best, scores[best])); chosen.append(best)
        print(f"  step {step}: add '{best}'  -> validation F1 {scores[best]:.2f}%   ({len(scores)} candidates tried this step)")
    # Save every candidate's validation score, so the whole search can be audited.
    step1.rename("validation_F1_%").to_csv("results/feature_search_all_candidates.csv", header=True)
    print("\nTop 10 single additions on validation data (step 1):\n" + step1.head(10).round(2).to_string())

    # Now report on fresh TEST worlds that played no part in the search, next to the main model.
    sets = {"Main model (4 features)": list(FEATURE_COLS)}
    for step, name, _ in history: sets[f"Greedy search, {step} addition(s)"] = list(FEATURE_COLS) + [h[1] for h in history[:step]]
    sets["All features at once (every base column and pair)"] = pool
    models = {k: fit(train, v, 200) for k, v in sets.items()}
    # A supervised ceiling (it uses the labels, so it is not a deployable model): it also receives the model's own score as an input.
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1).fit(
        np.column_stack([-models["Main model (4 features)"].decision_function(train[FEATURE_COLS]), train[pool]]), train["is_anomaly"])
    rows = []
    # Evaluate every model, the supervised ceiling and two fusion rules on each fresh test world.
    for d, s in zip(test, TEST_SEEDS):
        main_scores = -models["Main model (4 features)"].decision_function(d[FEATURE_COLS]); policy = basic_policy_flag(d).to_numpy().astype(bool)
        vol = float(policy.mean())
        for name, cols in sets.items():
            sc = -models[name].decision_function(d[cols])
            rows.append({"seed": s, "model": name, "budget": "5% review budget", **metr(d, topk(sc, BUDGET)), "roc_auc": roc_auc_score(d.is_anomaly, sc), "avg_precision": average_precision_score(d.is_anomaly, sc)})
            rows.append({"seed": s, "model": name, "budget": "same alert volume as Basic's policy (~6%)", **metr(d, topk(sc, vol))})
        stack = rf.predict_proba(np.column_stack([main_scores, d[pool]]))[:, 1]
        rows.append({"seed": s, "model": "Second stage with the model's SCORE as an input (supervised ceiling)", "budget": "same alert volume as Basic's policy (~6%)", **metr(d, topk(stack, vol))})
        rows.append({"seed": s, "model": "Fusion: model top 5% OR Basic policy", "budget": "union", **metr(d, topk(main_scores, BUDGET) | policy)})
        rows.append({"seed": s, "model": "Fusion: model top 7% AND Basic policy", "budget": "intersection", **metr(d, topk(main_scores, 0.07) & policy)})
        rows.append({"seed": s, "model": "Basic ZTA policy (for reference)", "budget": "fixed rule", **metr(d, policy)})
    res = pd.DataFrame(rows); res.round(4).to_csv("results/feature_search_test_results.csv", index=False)
    cols = ["detection_%", "cross_ward_%", "false_pos_%", "precision_%", "F1_%"]
    out = res.groupby(["budget", "model"], sort=False)[cols + ["roc_auc", "avg_precision"]].mean().round(2)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 70)
    print(f"\n=== TEST results on {len(TEST_SEEDS)} fresh worlds (nothing was tuned on them) ===\n" + out.to_string())
    # Winner's curse check: how much of the validation gain survives on test data the search never saw?
    best_test = res[(res.budget == "5% review budget") & (res.model == f"Greedy search, {STEPS} addition(s)")]["F1_%"].mean()
    print(f"\nWinner's curse check: the {STEPS}-feature search winner scored {history[-1][2]:.2f}% F1 on validation and {best_test:.2f}% on the test worlds "
          f"(main model: validation {base_val:.2f}%, test {res[(res.budget == '5% review budget') & (res.model == 'Main model (4 features)')]['F1_%'].mean():.2f}%).")
    print("Saved -> results/feature_search_all_candidates.csv, results/feature_search_test_results.csv")
