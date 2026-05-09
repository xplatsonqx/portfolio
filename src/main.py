from pathlib import Path
import platform
import subprocess
from data_loader import DataLoader
from dataset_column_classifier import ColumnClassifier
from numerical_analyzer import NumericalAnalyzer
from report_generator import ReportGenerator
from column_profiler import profile_columns

def main():
    # baza projektu
    base_dir = Path(__file__).resolve().parent.parent

    # ścieżki
    data_path = base_dir / "data" / "DrugSalesData.csv"
    report_path = base_dir / "EDA_report.html"

    # load data
    loader = DataLoader(data_path)
    df = loader.load()

    # profile columns (dtype + inferred role)
    col_profile = profile_columns(df)
    numerical_cols = col_profile.loc[col_profile["role"] == "numerical", "column"].tolist()
    categorical_cols = col_profile.loc[col_profile["role"] == "categorical", "column"].tolist()
    # include booleans in categorical-style section
    categorical_cols += col_profile.loc[col_profile["role"] == "boolean", "column"].tolist()
    datetime_cols = col_profile.loc[col_profile["role"] == "datetime", "column"].tolist()

    # analyze numerical data
    analyzer = NumericalAnalyzer(df, numerical_cols)
    summary = analyzer.summarize()
    corr_pearson = analyzer.correlations(method="pearson")
    corr_spearman = analyzer.correlations(method="spearman")

    # HTML report
    report = ReportGenerator(output_dir=base_dir)
    report_path = report.write_html_report(
        df=df,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        column_profile=col_profile,
        summary=summary,
        corr_pearson=corr_pearson,
        corr_spearman=corr_spearman,
        report_name=report_path.name,
    )

    # Open report quietly after run (F5 workflow), without terminal noise.
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.run(["open", str(report_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        elif system == "Windows":
            subprocess.run(["cmd", "/c", "start", "", str(report_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            subprocess.run(["xdg-open", str(report_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        # Keep execution silent and non-blocking if opener is unavailable.
        pass

if __name__ == "__main__":
    main()