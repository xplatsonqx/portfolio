from pathlib import Path
from data_loader import DataLoader
from dataset_column_classifier import ColumnClassifier
from numerical_analyzer import NumericalAnalyzer
from report_generator import ReportGenerator

import webbrowser


def main():
    # baza projektu
    base_dir = Path(__file__).resolve().parent.parent

    # ścieżki
    data_path = base_dir / "data" / "DrugSalesData.csv"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    # load data
    loader = DataLoader(data_path)
    df = loader.load()

    print(f"Dataset shape: {df.shape}")

    # classify columns
    classifier = ColumnClassifier(df)
    numerical_cols = classifier.get_numerical_columns()
    categorical_cols = classifier.get_categorical_columns()

    print(f"Numerical columns: {len(numerical_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")

    # analyze numerical data
    analyzer = NumericalAnalyzer(df, numerical_cols)
    summary = analyzer.summarize()
    corr_pearson = analyzer.correlations(method="pearson")
    corr_spearman = analyzer.correlations(method="spearman")

    # HTML report
    report = ReportGenerator(output_dir=output_dir)
    report_path = report.write_html_report(
        df=df,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        summary=summary,
        corr_pearson=corr_pearson,
        corr_spearman=corr_spearman,
    )

    print(f"Only output generated: {report_path.resolve()}")
    webbrowser.open(report_path.resolve().as_uri())

if __name__ == "__main__":
    main()