"""
Run the full pre-split cleaning + leakage-safe split pipeline on the real
CICIDS2017 dataset, and save the resulting train/val/test sets to
ml/data/processed/. This is the authoritative preprocessing step — all
later modeling reads from these saved files, not from raw/ directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders.cicids2017_loader import CICIDS2017Loader
from src.preprocessing.clean import clean_dataset
from src.preprocessing.split import filter_rare_multiclass, leakage_safe_split
from src.utils.config import load_config


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]

    raw_dir = ml_root / "data" / "raw" / "cicids2017"
    processed_dir = ml_root / config["paths"]["processed_dir"] / "cicids2017"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw CICIDS2017 data...")
    df = CICIDS2017Loader(raw_dir).load()
    print(f"Raw shape: {df.shape}")

    print("\nCleaning (dedup + invalid rows)...")
    df = clean_dataset(df)
    print(f"Cleaned shape: {df.shape}")

    min_samples = config["multiclass"]["min_class_samples"]
    print(f"\nFiltering rare multiclass labels (< {min_samples} samples)...")
    df_multiclass, excluded = filter_rare_multiclass(df, "Label", min_samples)
    print(f"Multiclass-eligible shape: {df_multiclass.shape}")
    print(f"Excluded classes: {excluded}")

    # ---- Binary task: uses the FULL cleaned dataset (rare classes stay in) ----
    print("\nSplitting for BINARY task (all classes retained)...")
    train_b, val_b, test_b = leakage_safe_split(
        df,
        label_col="Label",
        test_size=config["split"]["test_size"],
        val_size=config["split"]["val_size"],
        random_seed=config["random_seed"],
    )
    for name, split_df in [("train", train_b), ("val", val_b), ("test", test_b)]:
        split_df.to_csv(processed_dir / f"binary_{name}.csv", index=False)
    print(f"Saved binary_{{train,val,test}}.csv to {processed_dir}")

    # ---- Multiclass task: uses the filtered dataset (rare classes excluded) ----
    print("\nSplitting for MULTICLASS task (rare classes excluded)...")
    train_m, val_m, test_m = leakage_safe_split(
        df_multiclass,
        label_col="Label",
        test_size=config["split"]["test_size"],
        val_size=config["split"]["val_size"],
        random_seed=config["random_seed"],
    )
    for name, split_df in [("train", train_m), ("val", val_m), ("test", test_m)]:
        split_df.to_csv(processed_dir / f"multiclass_{name}.csv", index=False)
    print(f"Saved multiclass_{{train,val,test}}.csv to {processed_dir}")

    print("\nDone. All splits are leakage-safe: deduplicated and cleaned "
          "BEFORE splitting, with stratified sampling on the label column.")


if __name__ == "__main__":
    main()