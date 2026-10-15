"""Shared helpers for the testing scripts: finding the project root and the other pipeline scripts."""
import sys
from pathlib import Path


def find_root():
    """The project root is the nearest parent folder that contains both src/ and data/."""
    for p in Path(__file__).resolve().parents:
        if (p / "src").is_dir() and (p / "data").is_dir():
            return p
    raise SystemExit("Could not find the project root (a folder containing src/ and data/).")


ROOT = find_root()
TESTING = ROOT / "testing"
DATASETS = TESTING / "datasets"
COMPARISON = TESTING / "comparison"


def add_script_folders():
    """Let the testing scripts import the other pipeline scripts wherever they sit under src/."""
    skip = {"venv", "__pycache__", "data", "results", "models", "logs", "testing_data"}
    src = ROOT / "src"
    for p in [src] + [d for d in src.iterdir() if d.is_dir() and d.name not in skip and not d.name.startswith(".")]:
        if str(p) not in sys.path:
            sys.path.append(str(p))
