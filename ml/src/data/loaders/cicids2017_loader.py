"""
Loader for CICIDS2017.

The public distribution ships as multiple daily CSV files (e.g.
Monday-WorkingHours.pcap_ISCX.csv, Tuesday-..., etc). This loader
concatenates every .csv file found directly under raw_dir.

Encoding note: the three "Web Attack" label rows contain the Unicode
replacement character (U+FFFD) where a dash should be. This corruption
exists in the CIC-published source file itself — it is not an artifact
of how we read it, and no encoding choice can recover the original
character, since it was already lost before the file was published. We
read as UTF-8 (the correct encoding for these files) and then explicitly
replace the replacement character with a plain hyphen, matching CIC's
documented label names.
"""

import pandas as pd

from .base_loader import BaseDatasetLoader
from ..validators.validate import validate_cicids2017


class CICIDS2017Loader(BaseDatasetLoader):
    def load(self) -> pd.DataFrame:
        csv_files = sorted(self.raw_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No .csv files found in {self.raw_dir}. "
                f"Place the CICIDS2017 daily CSVs there — see ml/data/README.md."
            )

        frames = []
        for path in csv_files:
            df = pd.read_csv(path, low_memory=False, encoding="utf-8")
            # CICIDS2017 CSVs are known to have leading/trailing whitespace
            # in column names (e.g. ' Label'). Normalize before anything else.
            df.columns = df.columns.str.strip()
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)

        if "Label" in combined.columns:
            combined["Label"] = (
                combined["Label"]
                .astype(str)
                .str.strip()
                # U+FFFD (replacement character) appears in the source file
                # in place of an en-dash for "Web Attack" labels.
                .str.replace("\ufffd", "-", regex=False)
            )

        validate_cicids2017(combined)
        return combined