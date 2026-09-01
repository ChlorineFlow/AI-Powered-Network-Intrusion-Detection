"""
Validation functions that check a loaded pandas DataFrame against the
expected schema for its dataset, BEFORE it is used anywhere in the
pipeline. These raise SchemaValidationError with a clear message rather
than letting bad data fail silently downstream.
"""

import pandas as pd

from .schemas import (
    CICIDS2017_REQUIRED_COLUMNS,
    NSL_KDD_COLUMNS,
    UNSW_NB15_REQUIRED_COLUMNS,
)


class SchemaValidationError(Exception):
    """Raised when a raw dataset file does not match its expected schema."""


def validate_cicids2017(df: pd.DataFrame) -> None:
    missing = CICIDS2017_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"CICIDS2017 file is missing required columns: {sorted(missing)}"
        )
    if df.empty:
        raise SchemaValidationError("CICIDS2017 file has no rows.")


def validate_nsl_kdd(df: pd.DataFrame) -> None:
    if len(df.columns) != len(NSL_KDD_COLUMNS):
        raise SchemaValidationError(
            f"NSL-KDD file has {len(df.columns)} columns, "
            f"expected {len(NSL_KDD_COLUMNS)}."
        )
    if df.empty:
        raise SchemaValidationError("NSL-KDD file has no rows.")


def validate_unsw_nb15(df: pd.DataFrame) -> None:
    missing = UNSW_NB15_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"UNSW-NB15 file is missing required columns: {sorted(missing)}"
        )
    if df.empty:
        raise SchemaValidationError("UNSW-NB15 file has no rows.")