"""
check_continuity.py
Checks that the access log shows no gaps in recording while staff are working.

It reads data/access_logs.csv and data/staff.csv, works out the working hours from the staff
shifts, and then checks:
  1. Every calendar day in the period has events.
  2. Every working hour has events on every day.
  3. The longest gap between consecutive events inside working hours (overnight excluded).
  4. Events outside working hours are counted separately (off-hours anomalies, early starts),
     so night-time activity is not mistaken for normal coverage.
It prints PASS or FAIL and saves a day-by-hour heatmap with the working hours marked.

Note: gaps are measured within each calendar day, so a shift that crosses midnight has its
midnight gap ignored.

Usage (run from the project root):
    py src/check_continuity.py
    py src/check_continuity.py --max-gap-minutes 90
"""

import argparse
import os
import sys
import matplotlib
matplotlib.use("Agg")                      # draw to a file, no window needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def working_hours_from_staff(staff_df):
    """Hours of the day (0-23) covered by at least one staff member's shift."""
    hours = set()
    for start, end in zip(staff_df["shift_start"], staff_df["shift_end"]):
        length = int((end - start) % 24) or 24
        hours.update((int(start) + k) % 24 for k in range(length))
    return sorted(hours)


def check_daily_coverage(df):
    days = df["timestamp"].dt.normalize()
    daily = days.value_counts()
    expected = pd.date_range(days.min(), days.max(), freq="D")
    missing = [d.date() for d in expected if d not in daily.index]
    print("=== Daily coverage ===")
    print(f"Date range: {expected[0].date()} to {expected[-1].date()}  ({len(expected)} days)")
    print(f"Days with ZERO events: {len(missing)}" + (f"  -> {missing[:10]}" if missing else ""))
    print(f"Events per day: min {daily.min()}, max {daily.max()}")
    return missing


def check_working_hour_coverage(df, work_hours):
    d = df.assign(date=df["timestamp"].dt.normalize(), hour=df["timestamp"].dt.hour)
    counts = d[d["hour"].isin(work_hours)].groupby(["date", "hour"]).size().unstack(fill_value=0)
    all_days = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
    counts = counts.reindex(index=all_days, columns=work_hours, fill_value=0)
    empty = int((counts == 0).sum().sum())
    print("\n=== Working-hour coverage ===")
    print(f"Working hours (from staff shifts): {work_hours[0]:02d}:00-{work_hours[-1] + 1:02d}:00 "
          f"({len(work_hours)} hours)")
    print(f"Day-hour cells with ZERO events: {empty} of {counts.size}")
    return empty


def check_max_gap(df, work_hours):
    w = df[df["timestamp"].dt.hour.isin(work_hours)].copy()
    w["date"] = w["timestamp"].dt.normalize()
    gaps = w.sort_values("timestamp").groupby("date")["timestamp"].diff().dropna()
    print("\n=== Gaps between consecutive events (inside working hours) ===")
    print(f"Mean gap: {gaps.mean()}   Longest gap: {gaps.max()}")
    return gaps.max()


def report_out_of_hours(df, work_hours):
    out = df[~df["timestamp"].dt.hour.isin(work_hours)]
    print("\n=== Events outside working hours ===")
    print(f"{len(out)} events ({100 * len(out) / len(df):.1f}% of the log)")
    if "is_anomaly" in out.columns and len(out):
        print(f"  labelled anomalous: {int((out['is_anomaly'] == 1).sum())}")
        print(f"  labelled legitimate (e.g. early starts): {int((out['is_anomaly'] == 0).sum())}")


def plot_heatmap(df, work_hours, save_path):
    d = df.assign(date=df["timestamp"].dt.date, hour=df["timestamp"].dt.hour)
    pivot = d.pivot_table(index="date", columns="hour", values="user_id", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(columns=range(24), fill_value=0)       # show all 24 hours, even empty ones
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.2, linecolor="white",
                     cbar_kws={"label": "Number of access events"})
    if work_hours == list(range(work_hours[0], work_hours[-1] + 1)):
        ax.axvline(work_hours[0], color="red", linewidth=1.5)
        ax.axvline(work_hours[-1] + 1, color="red", linewidth=1.5)
    plt.title("Access log: events by day and hour\n"
              "(red lines mark working hours; pale cells = few or no events)")
    plt.xlabel("Hour of day")
    plt.ylabel("Date")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved heatmap -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the access log for recording gaps.")
    parser.add_argument("--file", default="data/access_logs.csv")
    parser.add_argument("--staff", default="data/staff.csv")
    parser.add_argument("--max-gap-minutes", type=float, default=60,
                        help="longest acceptable gap inside working hours (default 60)")
    args = parser.parse_args()

    try:
        logs = pd.read_csv(args.file)
        staff = pd.read_csv(args.staff)
    except FileNotFoundError as e:
        sys.exit(f"File not found: {e.filename}\nRun the generators first, from the project root.")
    logs["timestamp"] = pd.to_datetime(logs["timestamp"])

    print("=" * 66)
    print("AUDIT LOG CONTINUITY CHECK")
    print("=" * 66)
    work_hours = working_hours_from_staff(staff)
    missing_days = check_daily_coverage(logs)
    empty_cells = check_working_hour_coverage(logs, work_hours)
    longest = check_max_gap(logs, work_hours)
    report_out_of_hours(logs, work_hours)
    plot_heatmap(logs, work_hours, "results/continuity_heatmap.png")

    problems = []
    if missing_days:
        problems.append(f"{len(missing_days)} day(s) with no events")
    if empty_cells:
        problems.append(f"{empty_cells} working-hour cell(s) with no events")
    if longest > pd.Timedelta(minutes=args.max_gap_minutes):
        problems.append(f"a {longest} gap inside working hours (limit {args.max_gap_minutes:.0f} min)")
    print("\nVerdict:", "FAIL - " + "; ".join(problems) if problems
          else "PASS - recording is continuous throughout working hours on every day.")
    print("=" * 66)
    sys.exit(1 if problems else 0)
