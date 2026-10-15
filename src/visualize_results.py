"""
visualize_results.py
Generates seaborn visualizations comparing the three security conditions
(Traditional / Basic ZTA / Enhanced ZTA) across the five evaluation metrics,
using the REAL experiment results produced by run_experiment.py.

Run from the project root:
    py src/visualize_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="Set2")

CONDITION_ORDER = ["Traditional", "Basic ZTA", "Enhanced ZTA"]

METRICS = [
    ("MTTD", "Mean Time to Detect (seconds)"),
    ("MTTR", "Mean Time to Respond (seconds)"),
    ("detection_rate", "Detection Rate (%)"),
    ("false_positive_rate", "False-Positive Rate (%)"),
    ("records_exposed", "Records Exposed Before Containment"),
]


def plot_boxplots(df, save_dir="results"):
    """One boxplot per metric, comparing the three conditions."""
    for metric, label in METRICS:
        plt.figure(figsize=(6, 4.5))
        sns.boxplot(data=df, x="condition", y=metric, order=CONDITION_ORDER)
        sns.stripplot(data=df, x="condition", y=metric, order=CONDITION_ORDER,
                      color="black", alpha=0.3, size=3, jitter=True)
        plt.title(f"{label} by Security Condition")
        plt.xlabel("Security Condition")
        plt.ylabel(label)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/boxplot_{metric}.png", dpi=150)
        plt.close()
        print(f"Saved: {save_dir}/boxplot_{metric}.png")


def plot_boxplots_log_scale(df, save_dir="results"):
    """Same boxplots as above but with a log-scaled y-axis for MTTD and
    MTTR specifically, since Traditional's values (days) and Enhanced
    ZTA's values (seconds) differ by several orders of magnitude and are
    difficult to compare visually on a linear scale."""
    for metric, label in [("MTTD", "Mean Time to Detect (seconds, log scale)"),
                           ("MTTR", "Mean Time to Respond (seconds, log scale)")]:
        plt.figure(figsize=(6, 4.5))
        ax = sns.boxplot(data=df, x="condition", y=metric, order=CONDITION_ORDER)
        ax.set_yscale("log")
        sns.stripplot(data=df, x="condition", y=metric, order=CONDITION_ORDER,
                      color="black", alpha=0.3, size=3, jitter=True)
        plt.title(f"{label} by Security Condition")
        plt.xlabel("Security Condition")
        plt.ylabel(label)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/boxplot_{metric}_logscale.png", dpi=150)
        plt.close()
        print(f"Saved: {save_dir}/boxplot_{metric}_logscale.png")


def plot_bar_means(df, save_dir="results"):
    """Bar chart of mean values with error bars (95% CI), per metric."""
    for metric, label in METRICS:
        plt.figure(figsize=(6, 4.5))
        sns.barplot(data=df, x="condition", y=metric, order=CONDITION_ORDER,
                    errorbar=("ci", 95), capsize=0.15)
        plt.title(f"Mean {label} by Security Condition")
        plt.xlabel("Security Condition")
        plt.ylabel(f"Mean {label}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/barplot_{metric}.png", dpi=150)
        plt.close()
        print(f"Saved: {save_dir}/barplot_{metric}.png")


def plot_scenario_comparison(df, metric="MTTD", save_dir="results"):
    """Compares a metric across conditions, split by scenario (off-hours vs cross-ward)."""
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=df, x="condition", y=metric, hue="scenario", order=CONDITION_ORDER)
    plt.title(f"{metric} by Condition and Scenario")
    plt.xlabel("Security Condition")
    plt.ylabel(metric)
    plt.legend(title="Scenario")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/scenario_comparison_{metric}.png", dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/scenario_comparison_{metric}.png")


def plot_summary_table(df):
    """Prints descriptive statistics (mean, std, median) per condition per metric."""
    metric_cols = [m for m, _ in METRICS]
    summary = df.groupby("condition")[metric_cols].agg(["mean", "std", "median"]).round(2)
    summary = summary.reindex(CONDITION_ORDER)
    print("\n=== Descriptive Statistics by Condition ===")
    print(summary.to_string())
    return summary


if __name__ == "__main__":
    import os, sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results/experiment_results.csv"
    out = "results" if len(sys.argv) <= 1 else "results/charts_" + os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(path)

    plot_summary_table(df)
    plot_boxplots(df, out)
    plot_boxplots_log_scale(df, out)
    plot_bar_means(df, out)
    plot_scenario_comparison(df, metric="MTTD", save_dir=out)
    plot_scenario_comparison(df, metric="MTTR", save_dir=out)

    print(f"\nAll charts saved to the {out}/ folder.")
