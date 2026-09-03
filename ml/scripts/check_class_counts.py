"""
Check per-class training sample counts for NSL-KDD and UNSW-NB15
multiclass tasks, to decide whether the same min_class_samples
threshold applied to CICIDS2017 should also exclude any classes here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders.nsl_kdd_loader import NSLKDDLoader
from src.data.loaders.unsw_nb15_loader import UNSWNB15TrainTestLoader
from src.utils.config import load_config


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]
    threshold = config["multiclass"]["min_class_samples"]

    print(f"Configured min_class_samples threshold: {threshold}\n")

    nsl_train = NSLKDDLoader(ml_root / "data" / "raw" / "nsl_kdd", "KDDTrain+.txt").load()
    print("=" * 60)
    print("NSL-KDD training set class counts:")
    print("=" * 60)
    counts = nsl_train["label"].value_counts()
    print(counts)
    print(f"\nClasses below threshold ({threshold}): "
          f"{counts[counts < threshold].index.tolist()}")

    unsw_train = UNSWNB15TrainTestLoader(
        ml_root / "data" / "raw" / "unsw_nb15", "UNSW_NB15_training-set.csv"
    ).load()
    print("\n" + "=" * 60)
    print("UNSW-NB15 training set class counts (attack_cat):")
    print("=" * 60)
    counts = unsw_train["attack_cat"].value_counts()
    print(counts)
    print(f"\nClasses below threshold ({threshold}): "
          f"{counts[counts < threshold].index.tolist()}")


if __name__ == "__main__":
    main()