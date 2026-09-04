"""
Mirrors ml/src/preprocessing/common_features.py's extraction functions,
but operates on a single feature dict (as received via the API) rather
than a full DataFrame — used by the FastAPI routing/unified endpoints.
"""

import numpy as np
import pandas as pd

COMMON_FEATURE_NAMES = ["duration_sec", "src_bytes", "dst_bytes", "total_bytes", "log_duration"]


def build_common_features_row(duration_sec: float, src_bytes: float, dst_bytes: float) -> pd.DataFrame:
    """Builds the 5-feature common representation from 3 raw inputs
    that exist, in some form, across all three source datasets."""
    total_bytes = src_bytes + dst_bytes
    log_duration = np.log1p(max(duration_sec, 0))
    return pd.DataFrame([{
        "duration_sec": duration_sec,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "total_bytes": total_bytes,
        "log_duration": log_duration,
    }])