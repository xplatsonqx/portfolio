from pathlib import Path
from visualizer import Visualizer
from data_loader import DataLoader
from dataset_column_classifier import ColumnClassifier
from numeric_analyzer import NumericAnalyzer


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
    numeric_cols = classifier.get_numeric_columns()
    categorical_cols = classifier.get_categorical_columns()

    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")

    # analyze numeric data
    analyzer = NumericAnalyzer(df, numeric_cols)
    summary = analyzer.summarize()

    # save summary
    output_path = output_dir / "summary.csv"
    summary.to_csv(output_path)

    print(f"Summary saved to: {output_path.resolve()}")

    # plots
    plots_dir = output_dir / "plots"
    visualizer = Visualizer(plots_dir)

    for column in numeric_cols:
        visualizer.save_histogram(df[column], column)
        visualizer.save_boxplot(df[column], column)

    print(f"Plots saved to: {plots_dir.resolve()}")

if __name__ == "__main__":
    main()