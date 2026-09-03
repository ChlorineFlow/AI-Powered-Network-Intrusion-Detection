"""
Loader for UNSW-NB15.

The public distribution ships as multiple CSV parts
(UNSW-NB15_1.csv ... UNSW-NB15_4.csv) plus a features list file. This
loader concatenates every .csv file found directly under raw_dir and
normalizes label casing (the 'attack_cat' field is known to contain
inconsistent capitalization and stray whitespace across the four parts).
"""

import pandas as pd

from .base_loader import BaseDatasetLoader
from ..validators.validate import validate_unsw_nb15


class UNSWNB15Loader(BaseDatasetLoader):
    def load(self) -> pd.DataFrame:
        csv_files = sorted(self.raw_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No .csv files found in {self.raw_dir}. "
                f"Place the UNSW-NB15 CSV parts there — see ml/data/README.md."
            )

        frames = []
        for path in csv_files:
            df = pd.read_csv(path, low_memory=False)
            df.columns = df.columns.str.strip().str.lower()
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)

        if "attack_cat" in combined.columns:
            combined["attack_cat"] = (
                combined["attack_cat"]
                .astype(str)
                .str.strip()
                .replace({"nan": "Normal"})
            )

        validate_unsw_nb15(combined)
        return combined


class UNSWNB15TrainTestLoader(BaseDatasetLoader):
    """
    Loader for the official UNSW-NB15 training-set/testing-set CSVs
    (as published by the dataset's authors) — a fixed, pre-split format
    distinct from the raw 4-part CSVs handled by UNSWNB15Loader above.
    """

    def __init__(self, raw_dir, filename: str):
        super().__init__(raw_dir)
        self.filename = filename

    def load(self) -> pd.DataFrame:
        path = self.raw_dir / self.filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Place UNSW_NB15_training-set.csv / "
                f"UNSW_NB15_testing-set.csv in {self.raw_dir} — see ml/data/README.md."
            )
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        validate_unsw_nb15(df)
        return df