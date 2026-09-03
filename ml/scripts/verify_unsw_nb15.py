"""
Inspect the real UNSW-NB15 training/testing set files before building
any preprocessing pipeline on assumptions. Prints shape, columns, and
label distributions so we can confirm the actual structure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


def main():
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "unsw_nb15"

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {raw_dir}. Place the dataset files there first.")
        return

    for path in csv_files:
        print(f"\n{'=' * 70}\nFile: {path.name}\n{'=' * 70}")
        df = pd.read_csv(path, low_memory=False)
        print(f"Shape: {df.shape}")
        print(f"\nColumns ({len(df.columns)}):")
        print(list(df.columns))
        print(f"\nDtypes:")
        print(df.dtypes.value_counts())

        if "label" in df.columns:
            print(f"\n'label' value counts:")
            print(df["label"].value_counts())
        if "attack_cat" in df.columns:
            print(f"\n'attack_cat' value counts:")
            print(df["attack_cat"].value_counts())

        print(f"\nMissing values (top 5): ")
        print(df.isnull().sum().sort_values(ascending=False).head(5))


if __name__ == "__main__":
    main()