import pandas as pd


class ColumnClassifier:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_numerical_columns(self) -> list[str]:
        return self.df.select_dtypes(include="number").columns.tolist()

    def get_categorical_columns(self) -> list[str]:
        return self.df.select_dtypes(include="object").columns.tolist()