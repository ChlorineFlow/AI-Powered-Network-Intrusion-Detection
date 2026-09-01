"""
Quick manual check: load the real CICIDS2017 CSVs through our loader +
validator, and report basic stats. Run this once after placing the
dataset files, to confirm everything reads correctly before EDA begins.
"""

import sys
from pathlib import Path

# Make sure ml/ is on the path when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders.cicids2017_loader import CICIDS2017Loader


def main():
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "cicids2017"
    loader = CICIDS2017Loader(raw_dir)

    print(f"Loading CSVs from: {raw_dir}")
    df = loader.load()

    print("\n--- Load successful ---")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nColumn names (first 10): {list(df.columns[:10])}")

    if "Label" in df.columns:
        print("\nLabel value counts:")
        print(df["Label"].value_counts())
    else:
        print("\nWARNING: 'Label' column not found after validation — this "
              "should not happen since validation passed.")

    print("\nMissing values per column (top 10 by count):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    print("\nMemory usage:")
    print(f"{df.memory_usage(deep=True).sum() / (1024 ** 2):.1f} MB")


if __name__ == "__main__":
    main()