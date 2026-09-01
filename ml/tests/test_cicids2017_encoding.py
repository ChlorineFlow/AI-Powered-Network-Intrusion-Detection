"""
Regression test for the CICIDS2017 label-encoding fix.

Uses a tiny synthetic CSV written to a temp directory — not real dataset
rows — to confirm the loader correctly replaces the Unicode replacement
character (U+FFFD) with a hyphen in Label values, and leaves normal
labels untouched.
"""

import pandas as pd

from src.data.loaders.cicids2017_loader import CICIDS2017Loader


def test_replacement_character_is_fixed(tmp_path):
    # Build a minimal synthetic CSV matching the required schema, with a
    # corrupted label exactly as it appears in the real source files.
    df = pd.DataFrame({
        "Destination Port": [80, 443],
        "Flow Duration": [100, 200],
        "Total Fwd Packets": [1, 2],
        "Total Backward Packets": [1, 2],
        "Flow Bytes/s": [10.0, 20.0],
        "Flow Packets/s": [1.0, 2.0],
        "Label": ["BENIGN", "Web Attack \ufffd Brute Force"],
    })
    csv_path = tmp_path / "synthetic.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    loader = CICIDS2017Loader(tmp_path)
    result = loader.load()

    labels = result["Label"].tolist()
    assert "BENIGN" in labels
    assert "Web Attack - Brute Force" in labels
    assert not any("\ufffd" in label for label in labels)