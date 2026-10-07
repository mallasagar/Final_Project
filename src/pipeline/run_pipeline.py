import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
LOG_DIR = PROJECT_ROOT / "logs"

# (script name, extra command-line arguments, short description)
# Use the bare file name, e.g. "patient_loader.py". The runner finds it anywhere under src/.
CORE_STEPS = [
    ("patient_loader.py", [], "assign Synthea patients to wards"),
    ("staff_generator.py", [], "generate staff"),
    ("validate_staff.py", [], "validate staff"),
    ("validate_patient_wards.py", [], "validate patient-ward mapping"),
    ("access_log_generator.py", [], "generate the access log"),
    ("validate_access_logs.py", [], "validate the access log"),
    ("check_duplicates.py", [], "check for duplicate events"),
    ("check_continuity.py", [], "check recording continuity"),
    ("feature_engineering.py", [], "engineer features"),
    ("validate_features.py", [], "validate features"),
    ("data_quality_report.py", [], "data-quality audit of the features"),
    ("anomaly_model.py", [], "train the Isolation Forest"),
    ("orchestration.py", [], "risk tiers and automated responses"),
    ("run_experiment.py", [], "three-condition experiment (resampled)"),
    ("run_experiment_independent.py", [], "three-condition experiment (independent datasets)"),
    ("statistical_analysis.py", ["results/experiment_results_independent.csv"], "statistics on the independent results"),
    ("test_on_holdout_multi.py", [], "held-out generalisation test"),
    ("make_figures.py", [], "thesis figures"),
    ("make_summary_figure.py", [], "one-page summary figure for the supervisor"),
    ("show_results.py", [], "print the headline results"),
]
ANALYSIS_STEPS = [
    ("contamination_sweep.py", [], "contamination trade-off table"),
    ("noise_sensitivity.py", [], "noise sensitivity analysis"),
    ("operating_point_analysis.py", [], "cost-based choice of the cut-off"),
    ("evaluate_new_datasets.py", [], "test on 100 newly created datasets"),
]
OUTPUT_DIRS = ["data", "data/csv", "models", "results", "results/figures"]
REQUIRED_INPUT = PROJECT_ROOT / "data" / "csv" / "patients.csv"


class Logger:
    def __init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = LOG_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"

    def __call__(self, message="", stamp=True):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}" if stamp and message else message
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def find_script(name):
    """Locate a script by file name anywhere under src/. Returns (path, error message)."""
    name = name.replace("\\", "/").lstrip("/")          # tolerate "/pipeline/x.py" and "pipeline\\x.py"
    matches = [p for p in SRC.rglob(name) if "__pycache__" not in p.parts]
    if not matches:
        return None, f"not found anywhere under {SRC}"
    if len(matches) > 1:
        listed = "; ".join(str(p.relative_to(PROJECT_ROOT)) for p in matches)
        return None, f"{len(matches)} copies found, so it is unclear which one to run: {listed}. Delete the stale one."
    return matches[0], None


def run_script(script, args, log):
    """Run one script, streaming its output to the console and the log. Returns (ok, seconds)."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    start = time.perf_counter()
    try:
        proc = subprocess.Popen([sys.executable, str(script), *args], cwd=PROJECT_ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            log("    " + line.rstrip(), stamp=False)
        proc.wait()
        return proc.returncode == 0, time.perf_counter() - start, proc.returncode
    except Exception as error:                                    # e.g. the interpreter could not start
        log(f"ERROR while starting {script.name}: {error}")
        return False, time.perf_counter() - start, -1


def freeze(log):
    target = PROJECT_ROOT / "frozen" / f"{datetime.now():%Y%m%d_%H%M%S}"
    for folder in ["data", "models", "results"]:
        if (PROJECT_ROOT / folder).exists():
            shutil.copytree(PROJECT_ROOT / folder, target / folder)
    log(f"Frozen copy of data, models and results saved to {target}")


def main():
    parser = argparse.ArgumentParser(description="Run the pipeline end to end.")
    parser.add_argument("--list", action="store_true", help="show the steps and exit")
    parser.add_argument("--from", dest="start", help="start at this step (name or number)")
    parser.add_argument("--only", help="run only this step (name or number)")
    parser.add_argument("--with-analysis", action="store_true", help="also run the slower sensitivity analyses")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed step")
    parser.add_argument("--freeze", action="store_true", help="copy data, models and results when finished")
    args = parser.parse_args()

    steps = CORE_STEPS + (ANALYSIS_STEPS if args.with_analysis else [])
    names = [s[0] for s in steps]

    def resolve(ref):
        if ref.isdigit() and 1 <= int(ref) <= len(steps):
            return int(ref) - 1
        for i, n in enumerate(names):
            if n == ref or Path(n).name in (ref, ref + ".py"):
                return i
        sys.exit(f"Unknown step '{ref}'. Use --list to see the steps.")

    if args.list:
        for i, (n, a, d) in enumerate(steps, 1):
            print(f"{i:>2}. {n:<34}{d}")
        return 0

    selected = list(enumerate(steps, 1))
    if args.only:
        selected = [selected[resolve(args.only)]]
    elif args.start:
        selected = selected[resolve(args.start):]

    log = Logger()
    t0 = time.perf_counter()
    log("=" * 74, stamp=False)
    log("STARTING PIPELINE")
    log(f"Python: {sys.executable}")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Log file: {log.path}")

    # ---- pre-flight checks: fail early, with a clear message ----
    problems = []
    if not REQUIRED_INPUT.exists() and any(n == "patient_loader.py" for _, (n, _, _) in selected):
        problems.append(f"Missing input: {REQUIRED_INPUT} (the Synthea patients.csv)")
    for _, (n, _, _) in selected:
        path, err = find_script(n)
        if err:
            problems.append(f"{n}: {err}")
    if problems:
        log("PRE-FLIGHT CHECK FAILED:")
        for p in problems:
            log("  - " + p)
        return 2
    for d in OUTPUT_DIRS:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
    log(f"Steps to run: {len(selected)}")

    # ---- run ----
    results, failed_step = [], None
    for number, (name, extra, desc) in selected:
        script, _ = find_script(name)
        log("=" * 74, stamp=False)
        log(f"STEP {number}/{len(steps)}: {name}  -  {desc}")
        ok, secs, code = run_script(script, extra, log)
        results.append((number, name, "OK" if ok else f"FAILED (exit {code})", secs))
        log(f"{'SUCCESS' if ok else 'FAILED'}: {name} ({secs:.1f} s)")
        if not ok and failed_step is None:
            failed_step = name
        if not ok and not args.keep_going:
            log("Stopping: later steps depend on this one. Use --keep-going to continue anyway.")
            break

    # ---- summary ----
    log("=" * 74, stamp=False)
    log("SUMMARY")
    for number, name, status, secs in results:
        log(f"  {number:>2}. {name:<34}{status:<18}{secs:>7.1f} s")
    skipped = len(selected) - len(results)
    if skipped:
        log(f"  ({skipped} later step(s) not run)")
    total = time.perf_counter() - t0
    if failed_step:
        log(f"PIPELINE FAILED at {failed_step}. Total time {total:.1f} s. Full output: {log.path}")
        return 1
    log(f"PIPELINE COMPLETED SUCCESSFULLY. {len(results)} step(s), {total:.1f} s. Full output: {log.path}")
    if args.freeze:
        freeze(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
