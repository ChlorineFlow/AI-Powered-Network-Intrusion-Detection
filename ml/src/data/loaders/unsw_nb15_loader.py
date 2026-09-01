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