"""
Extracts a small, honestly-limited common feature subset shared across
CICIDS2017, NSL-KDD, and UNSW-NB15 — for the exploratory unified
binary classification experiment.

IMPORTANT LIMITATION, stated upfront: NSL-KDD does not provide
per-connection packet counts (its 'count'/'srv_count' fields are
time-window connection aggregates, not packet counts within a single
flow — a fundamentally different concept from CICIDS2017's
Total Fwd/Backward Packets or UNSW-NB15's spkts/dpkts). Packet-count
features are therefore EXCLUDED from the common set entirely, rather
than approximated or imputed, since a fabricated/imputed value would
misrepresent what the model is actually learning from.

The resulting common feature set is intentionally small (duration and
byte counts only) — this is a disclosed constraint of attempting
unification across three datasets with fundamentally different native
feature schemas, not an oversight.
"""

import numpy as np
import pandas as pd

COMMON_FEATURE_NAMES = [
    "duration_sec",
    "src_bytes",
    "dst_bytes",
    "total_bytes",       # derived: src_bytes + dst_bytes
    "log_duration",      # derived: log1p(duration_sec), to tame scale
]


def extract_cicids2017_common(df: pd.DataFrame) -> pd.DataFrame:
    """CICIDS2017's Flow Duration is in microseconds; convert to seconds
    for consistency with the other two datasets."""
    out = pd.DataFrame()
    out["duration_sec"] = df["Flow Duration"] / 1_000_000.0
    out["src_bytes"] = df["Total Length of Fwd Packets"]
    out["dst_bytes"] = df["Total Length of Bwd Packets"]
    out["total_bytes"] = out["src_bytes"] + out["dst_bytes"]
    out["log_duration"] = np.log1p(out["duration_sec"].clip(lower=0))
    return out


def extract_nsl_kdd_common(df: pd.DataFrame) -> pd.DataFrame:
    """NSL-KDD's duration and byte fields are already in seconds/bytes."""
    out = pd.DataFrame()
    out["duration_sec"] = df["duration"]
    out["src_bytes"] = df["src_bytes"]
    out["dst_bytes"] = df["dst_bytes"]
    out["total_bytes"] = out["src_bytes"] + out["dst_bytes"]
    out["log_duration"] = np.log1p(out["duration_sec"].clip(lower=0))
    return out


def extract_unsw_nb15_common(df: pd.DataFrame) -> pd.DataFrame:
    """UNSW-NB15's dur/sbytes/dbytes are already in seconds/bytes."""
    out = pd.DataFrame()
    out["duration_sec"] = df["dur"]
    out["src_bytes"] = df["sbytes"]
    out["dst_bytes"] = df["dbytes"]
    out["total_bytes"] = out["src_bytes"] + out["dst_bytes"]
    out["log_duration"] = np.log1p(out["duration_sec"].clip(lower=0))
    return out