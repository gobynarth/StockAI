import os
import sys
from pathlib import Path


BASE_DIR = Path(os.environ.get("STOCKAI_BASE", Path(__file__).resolve().parent))
KRONOS_DIR = Path(os.environ.get("KRONOS_PATH", BASE_DIR.parent / "Kronos"))


def add_kronos_to_path():
    kronos = str(KRONOS_DIR)
    if kronos not in sys.path:
        sys.path.append(kronos)


def base_path(*parts):
    return str(BASE_DIR.joinpath(*parts))
