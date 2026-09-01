"""Shared interface for all dataset loaders."""

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class BaseDatasetLoader(ABC):
    """Every dataset loader implements load() and returns a validated
    pandas DataFrame. Loaders never modify files in raw_dir — reading
    only."""

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)
        if not self.raw_dir.exists():
            raise FileNotFoundError(
                f"Raw data directory not found: {self.raw_dir}. "
                f"See ml/data/README.md for download instructions."
            )

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load, lightly clean column names, validate schema, and return
        the raw DataFrame (no feature engineering here)."""
        raise NotImplementedError