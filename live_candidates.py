from pathlib import Path

import pandas as pd


LIVE_CANDIDATES_FILE = "approved_live_candidates.csv"
PENDING_CANDIDATES_FILE = "approved_pending_candidates.csv"
BOOL_COLUMNS = ("skip_low_vix", "skip_near_earnings", "allow_short")
FLOAT_COLUMNS = ("temp", "oos_acc", "alloc", "tp", "sl", "trailing_pct")
INT_COLUMNS = ("horizon", "lookback")


def _read_candidate_csv(base_dir, filename):
    path = Path(base_dir) / filename
    return pd.read_csv(path)


def load_live_candidates(base_dir):
    df = _read_candidate_csv(base_dir, LIVE_CANDIDATES_FILE)
    active = {}
    for _, row in df.iterrows():
        cfg = {"tier": str(row["tier"])}
        for key in INT_COLUMNS:
            cfg[key] = int(row[key])
        for key in FLOAT_COLUMNS:
            cfg[key] = float(row[key])
        for key in BOOL_COLUMNS:
            cfg[key] = bool(row.get(key, False))
        active[str(row["ticker"])] = cfg
    return active


def load_pending_candidates(base_dir):
    df = _read_candidate_csv(base_dir, PENDING_CANDIDATES_FILE)
    return df.to_dict("records")
