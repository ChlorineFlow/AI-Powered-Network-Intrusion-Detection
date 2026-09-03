"""
Quick manual check: load the real NSL-KDD files through our loader +
validator, and report basic stats. Mirrors verify_cicids2017.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders.nsl_kdd_loader import NSLKDDLoader


def main():
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "nsl_kdd"

    for filename in ["KDDTrain+.txt", "KDDTest+.txt"]:
        print(f"\n{'=' * 70}\nLoading: {filename}\n{'=' * 70}")
        loader = NSLKDDLoader(raw_dir, filename=filename)
        df = loader.load()

        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print(f"\nLabel value counts (top 15):")
        print(df["label"].value_counts().head(15))
        print(f"\nProtocol type distribution:")
        print(df["protocol_type"].value_counts())
        print(f"\nMissing values: {df.isnull().sum().sum()}")


if __name__ == "__main__":
    main()