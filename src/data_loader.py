
import pandas as pd


class DataLoader:
    def __init__(self, path: str, separator: str = ";"):
        self.path = path
        self.separator = separator

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path, sep=self.separator, low_memory=False)