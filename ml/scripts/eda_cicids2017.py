"""
Exploratory Data Analysis for CICIDS2017.

Run after the dataset is placed in ml/data/raw/cicids2017/. Produces
printed statistics and saves figures to ml/figures/eda/. All numbers
here come from the real loaded dataset — nothing is fabricated or
estimated.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save PNGs
import matplotlib.pyplot as plt

from src.data.loaders.cicids2017_loader import CICIDS2017Loader

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures" / "eda"


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "cicids2017"
    df = CICIDS2017Loader(raw_dir).load()

    section("1. Dataset shape")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # -----------------------------------------------------------------
    section("2. Binary label distribution (BENIGN vs ATTACK)")
    binary = df["Label"].apply(lambda x: "BENIGN" if x == "BENIGN" else "ATTACK")
    binary_counts = binary.value_counts()
    print(binary_counts)
    print(f"\nAttack ratio: {binary_counts.get('ATTACK', 0) / len(df):.4%}")

    plt.figure(figsize=(6, 4))
    binary_counts.plot(kind="bar", color=["#2b8a3e", "#c92a2a"])
    plt.title("Binary class distribution: BENIGN vs ATTACK")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "binary_class_distribution.png", dpi=120)
    plt.close()

    # -----------------------------------------------------------------
    section("3. Multiclass label distribution")
    class_counts = df["Label"].value_counts()
    print(class_counts)

    plt.figure(figsize=(10, 6))
    class_counts.plot(kind="barh", logx=True)
    plt.title("Multiclass label distribution (log scale)")
    plt.xlabel("Count (log scale)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "multiclass_distribution.png", dpi=120)
    plt.close()

    # -----------------------------------------------------------------
    section("4. Missing values")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(missing)
    else:
        print("No missing (NaN) values found.")

    # -----------------------------------------------------------------
    section("5. Infinite values (known CICIDS2017 issue in rate columns)")
    numeric_df = df.select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_df).sum()
    inf_counts = inf_counts[inf_counts > 0].sort_values(ascending=False)
    if len(inf_counts) > 0:
        print(inf_counts)
    else:
        print("No infinite values found.")

    # -----------------------------------------------------------------
    section("6. Duplicate rows")
    dup_count = df.duplicated().sum()
    print(f"Exact duplicate rows: {dup_count:,} ({dup_count / len(df):.2%} of dataset)")

    # -----------------------------------------------------------------
    section("7. Numeric feature summary statistics (first 10 numeric cols)")
    print(numeric_df.iloc[:, :10].describe().T)

    # -----------------------------------------------------------------
    section("8. Class imbalance ratio")
    majority = class_counts.max()
    minority = class_counts.min()
    print(f"Majority class: {class_counts.idxmax()} ({majority:,})")
    print(f"Minority class: {class_counts.idxmin()} ({minority:,})")
    print(f"Imbalance ratio (majority:minority): {majority / minority:,.1f} : 1")

    # -----------------------------------------------------------------
    section("9. Correlation heatmap (subset of numeric features)")
    # Full 78-feature correlation matrix is unreadable as a plot; sample
    # a manageable subset of commonly-referenced features instead.
    candidate_cols = [
        c for c in [
            "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
            "Flow Bytes/s", "Flow Packets/s", "Fwd Packet Length Mean",
            "Bwd Packet Length Mean", "Destination Port",
        ] if c in numeric_df.columns
    ]
    if candidate_cols:
        corr = numeric_df[candidate_cols].corr()
        plt.figure(figsize=(8, 6))
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(label="Correlation")
        plt.xticks(range(len(candidate_cols)), candidate_cols, rotation=90)
        plt.yticks(range(len(candidate_cols)), candidate_cols)
        plt.title("Correlation heatmap (selected features)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=120)
        plt.close()
        print(f"Saved to {FIGURES_DIR / 'correlation_heatmap.png'}")

    section("Done")
    print(f"Figures saved under: {FIGURES_DIR}")


if __name__ == "__main__":
    main()