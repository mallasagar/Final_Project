"""
run_testing.py  -  create datasets, test the model on each, and compare them, in one command.

    py src/testing/run_testing.py --profiles --clean         # one dataset per stress profile
    py src/testing/run_testing.py --n 10 --clean             # 10 datasets with the design settings
    py src/testing/run_testing.py --profiles --repeats 3     # three datasets per profile
Everything lands in the new testing/ folder at the project root:
    testing/datasets/<one folder per dataset>/    access_logs.csv, config.json, access_logs_scored.csv, results.csv
    testing/comparison/                           the side-by-side comparison, report and figures
"""
import subprocess
import sys
from pathlib import Path

import _common as C

HERE = Path(__file__).resolve().parent


def run(script, args):
    print("=" * 80 + f"\n{script} {' '.join(args)}\n" + "=" * 80, flush=True)
    code = subprocess.run([sys.executable, str(HERE / script), *args], cwd=C.ROOT).returncode
    if code != 0:
        raise SystemExit(f"{script} failed (exit code {code}). Stopping.")


if __name__ == "__main__":
    passthrough = sys.argv[1:]                       # --n, --profiles, --repeats, --start-seed, --days, --clean
    run("create_datasets.py", passthrough)
    run("test_datasets.py", [])
    run("compare_datasets.py", [])
