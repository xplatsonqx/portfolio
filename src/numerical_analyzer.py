import pandas as pd


class NumericalAnalyzer:
    def __init__(self, df: pd.DataFrame, numerical_columns: list[str]):
        self.df = df
        self.numerical_columns = numerical_columns

    def summarize(self) -> pd.DataFrame:
        df_num = self.df[self.numerical_columns]
        summary = df_num.describe().T

        n_rows = len(df_num)

        missing_values = df_num.isnull().sum()
        summary["missing_values"] = missing_values
        summary["missing_pct"] = (missing_values / n_rows * 100.0) if n_rows else 0.0

        unique = df_num.nunique(dropna=True)
        summary["unique"] = unique
        summary["unique_pct"] = (unique / n_rows * 100.0) if n_rows else 0.0

        zeros = (df_num == 0).sum(numeric_only=True)
        summary["zeros"] = zeros
        summary["zeros_pct"] = (zeros / n_rows * 100.0) if n_rows else 0.0

        summary["variance"] = df_num.var()
        summary["skew"] = df_num.skew()
        summary["kurtosis"] = df_num.kurtosis()

        q1 = df_num.quantile(0.25)
        q3 = df_num.quantile(0.75)
        iqr = q3 - q1
        summary["q1"] = q1
        summary["q3"] = q3
        summary["iqr"] = iqr

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = df_num.lt(lower) | df_num.gt(upper)
        outlier_count = outlier_mask.sum()
        summary["outlier_count"] = outlier_count
        summary["outlier_pct"] = (outlier_count / n_rows * 100.0) if n_rows else 0.0

        return summary

    def correlations(self, method: str = "pearson") -> pd.DataFrame:
        df_num = self.df[self.numerical_columns]
        return df_num.corr(method=method)
