"""
Unit tests for schema validators, using small synthetic DataFrames —
NOT real dataset samples. These test the validation logic itself, not
model performance, so they can run without any downloaded data.
"""

import pandas as pd
import pytest

from src.data.validators.validate import (
    validate_cicids2017,
    validate_nsl_kdd,
    validate_unsw_nb15,
    SchemaValidationError,
)
from src.data.validators.schemas import NSL_KDD_COLUMNS


def test_cicids2017_valid():
    df = pd.DataFrame({
        "Destination Port": [80],
        "Flow Duration": [100],
        "Total Fwd Packets": [1],
        "Total Backward Packets": [1],
        "Flow Bytes/s": [10.0],
        "Flow Packets/s": [1.0],
        "Label": ["BENIGN"],
    })
    validate_cicids2017(df)  # should not raise


def test_cicids2017_missing_column():
    df = pd.DataFrame({"Flow Duration": [100]})
    with pytest.raises(SchemaValidationError):
        validate_cicids2017(df)


def test_nsl_kdd_valid():
    df = pd.DataFrame([[0] * len(NSL_KDD_COLUMNS)], columns=NSL_KDD_COLUMNS)
    validate_nsl_kdd(df)  # should not raise


def test_nsl_kdd_wrong_column_count():
    df = pd.DataFrame([[0, 1, 2]], columns=["a", "b", "c"])
    with pytest.raises(SchemaValidationError):
        validate_nsl_kdd(df)


def test_unsw_nb15_valid():
    df = pd.DataFrame({
        "dur": [1.0], "proto": ["tcp"], "service": ["http"], "state": ["FIN"],
        "spkts": [1], "dpkts": [1], "sbytes": [100], "dbytes": [100],
        "rate": [1.0], "sttl": [64], "dttl": [64],
        "attack_cat": ["Normal"], "label": [0],
    })
    validate_unsw_nb15(df)  # should not raise


def test_unsw_nb15_missing_column():
    df = pd.DataFrame({"dur": [1.0]})
    with pytest.raises(SchemaValidationError):
        validate_unsw_nb15(df)