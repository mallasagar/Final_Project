"""
statistical_analysis.py
Performs the statistical comparison specified in Chapter 3.6: for each of
the five evaluation metrics, runs a Kruskal-Wallis H test across the three
security conditions, computes an effect size (epsilon-squared), and - where
the result is significant - runs post-hoc pairwise comparisons (Dunn's
test with Bonferroni correction) to identify which specific conditions
differ from one another.

Run from the project root:
    py src/statistical_analysis.py
"""

import pandas as pd
import numpy as np
from scipy.stats import kruskal
import scikit_posthocs as sp

METRICS = ["MTTD", "MTTR", "detection_rate", "false_positive_rate", "records_exposed"]
CONDITIONS = ["Traditional", "Basic ZTA", "Enhanced ZTA"]


def epsilon_squared(h_stat, n, k):
    """
    Effect size for the Kruskal-Wallis test.
    epsilon^2 = (H - k + 1) / (n - k)
    Interpretation (Tomczak & Tomczak, 2014, commonly cited convention):
        < 0.01  negligible
        < 0.06  small
        < 0.14  medium
        >= 0.14 large
    """
    return (h_stat - k + 1) / (n - k)


def interpret_effect_size(eps2):
    if eps2 < 0.01:
        return "negligible"
    elif eps2 < 0.06:
        return "small"
    elif eps2 < 0.14:
        return "medium"
    else:
        return "large"


def run_kruskal_wallis(df, metric, scenario=None):
    """Runs Kruskal-Wallis for one metric, optionally restricted to one
    scenario. Returns a dict of results."""
    subset = df if scenario is None else df[df["scenario"] == scenario]
    subset = subset.dropna(subset=[metric])

    groups = [subset[subset["condition"] == c][metric].values for c in CONDITIONS]

    # Kruskal-Wallis requires at least 2 non-empty groups with variation
    if any(len(g) == 0 for g in groups):
        return None

    h_stat, p_value = kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eps2 = epsilon_squared(h_stat, n, k)

    result = {
        "metric": metric,
        "scenario": scenario if scenario else "combined",
        "H": h_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "epsilon_squared": eps2,
        "effect_size": interpret_effect_size(eps2),
        "n_total": n,
    }
    return result, subset


def run_posthoc(subset, metric):
    """Dunn's test with Bonferroni correction for pairwise comparisons."""
    posthoc = sp.posthoc_dunn(
        subset, val_col=metric, group_col="condition", p_adjust="bonferroni"
    )
    return posthoc


def print_result(result):
    sig_marker = "***SIGNIFICANT***" if result["significant"] else "not significant"
    print(f"\n--- {result['metric']} ({result['scenario']}) ---")
    print(f"H = {result['H']:.3f}, p = {result['p_value']:.6f}  [{sig_marker}]")
    print(f"Effect size (epsilon-squared) = {result['epsilon_squared']:.4f} ({result['effect_size']})")


if __name__ == "__main__":
    import os, sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results/experiment_results.csv"
    tag = "" if len(sys.argv) <= 1 else "_" + os.path.splitext(os.path.basename(path))[0]
    print("Input file:", path)
    df = pd.read_csv(path)

    print("=" * 70)
    print("STATISTICAL ANALYSIS: Kruskal-Wallis H Test Across Security Conditions")
    print("=" * 70)

    all_results = []

    for metric in METRICS:
        # --- Combined analysis (both scenarios together) ---
        outcome = run_kruskal_wallis(df, metric)
        if outcome is None:
            print(f"\n--- {metric} (combined) --- SKIPPED (insufficient data)")
            continue

        result, subset = outcome
        print_result(result)
        all_results.append(result)

        if result["significant"]:
            print("\nPost-hoc pairwise comparisons (Dunn's test, Bonferroni-adjusted p-values):")
            posthoc = run_posthoc(subset, metric)
            print(posthoc.round(4).to_string())

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df[["metric", "scenario", "H", "p_value", "significant", "epsilon_squared", "effect_size"]]
    print(summary_df.round(4).to_string(index=False))

    summary_df.to_csv(f"results/statistical_summary{tag}.csv", index=False)
    print(f"\nSaved -> results/statistical_summary{tag}.csv")
