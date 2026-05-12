from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, sep: str = ";") -> pd.DataFrame:
    return pd.read_csv(path, sep=sep, low_memory=False)
