import pandas as pd

class NumericAnalyzer:
    def __init__(self, df: pd.DataFrame, numeric_columns: list[str]):
        self.df = df
        self.numeric_columns = numeric_columns

    def summarize(self) -> pd.DataFrame:
        summary = self.df[self.numeric_columns].describe().T

        summary["missing_values"] = self.df[self.numeric_columns].isnull().sum()

        return summary