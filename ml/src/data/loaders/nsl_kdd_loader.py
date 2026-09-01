"""
Loader for NSL-KDD.

KDDTrain+.txt and KDDTest+.txt have NO header row and are comma-separated.
Column names are supplied explicitly (see schemas.NSL_KDD_COLUMNS).
"""

import pandas as pd

from .base_loader import BaseDatasetLoader
from ..validators.schemas import NSL_KDD_COLUMNS
from ..validators.validate import validate_nsl_kdd


class NSLKDDLoader(BaseDatasetLoader):
    def __init__(self, raw_dir, filename: str = "KDDTrain+.txt"):
        super().__init__(raw_dir)
        self.filename = filename

    def load(self) -> pd.DataFrame:
        path = self.raw_dir / self.filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Place KDDTrain+.txt / KDDTest+.txt in "
                f"{self.raw_dir} — see ml/data/README.md."
            )

        df = pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS)
        validate_nsl_kdd(df)
        return df