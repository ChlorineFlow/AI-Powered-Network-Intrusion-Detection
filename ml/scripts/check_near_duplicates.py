"""
Check for near-duplicate rows between train and test splits — rows that
are not EXACT duplicates (already removed in Step 4) but are so similar
that they likely originate from the same attack burst (e.g. the same
DoS tool firing near-identical requests within milliseconds).

Method: round all numeric feature values to a coarser precision, then
treat matching rounded rows as "near-duplicates." This is intentionally
approximate — it directly tests the specific leakage hypothesis raised
in docs/research/methodology.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils.config import load_config

ROUNDING_DECIMALS = 1  # coarser rounding = stricter "near-duplicate" definition


def make_fingerprint(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    rounded = df[feature_cols].round(ROUNDING_DECIMALS)
    return rounded.astype(str).agg("|".join, axis=1)


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]
    processed_dir = ml_root / config["paths"]["processed_dir"] / "cicids2017"

    print("Loading binary train/test splits...")
    train_df = pd.read_csv(processed_dir / "binary_train.csv")
    test_df = pd.read_csv(processed_dir / "binary_test.csv")

    feature_cols = [c for c in train_df.columns if c != "Label"]
    numeric_cols = train_df[feature_cols].select_dtypes(include="number").columns.tolist()

    print(f"Fingerprinting {len(train_df):,} train rows and "
          f"{len(test_df):,} test rows using {len(numeric_cols)} numeric "
          f"features rounded to {ROUNDING_DECIMALS} decimal(s)...")

    train_fp = set(make_fingerprint(train_df, numeric_cols))
    test_fp = make_fingerprint(test_df, numeric_cols)

    overlap_mask = test_fp.isin(train_fp)
    overlap_count = overlap_mask.sum()
    overlap_pct = overlap_count / len(test_df)

    print(f"\n{'=' * 60}")
    print("NEAR-DUPLICATE OVERLAP RESULT")
    print(f"{'=' * 60}")
    print(f"Test rows with a near-identical match in train: "
          f"{overlap_count:,} / {len(test_df):,} ({overlap_pct:.2%})")

    # Break down by label — is the overlap concentrated in attack rows
    # (consistent with the "attack tool burst" hypothesis) or spread evenly?
    print("\nOverlap rate by class:")
    test_df["_is_near_dup"] = overlap_mask.values
    print(test_df.groupby("Label")["_is_near_dup"].mean().sort_values(ascending=False))

    print(f"\n{'=' * 60}")
    if overlap_pct > 0.05:
        print("SIGNIFICANT overlap found (>5%). This supports the concern "
              "that near-duplicate attack-burst flows are split across "
              "train/test, inflating reported performance beyond what "
              "would hold on genuinely novel traffic.")
    else:
        print("Overlap is low (<5%). This weighs AGAINST the near-duplicate "
              "leakage hypothesis being a major driver of the high reported "
              "performance — though it does not fully rule it out, since "
              "rounding to 1 decimal is still a coarse approximation.")


if __name__ == "__main__":
    main()