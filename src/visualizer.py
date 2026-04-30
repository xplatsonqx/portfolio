from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_histogram(self, series: pd.Series, column_name: str) -> Path:
        clean_series = series.dropna()

        plt.figure(figsize=(10, 6))
        plt.hist(clean_series, bins=30)
        plt.title(f"Distribution of {column_name}")
        plt.xlabel(column_name)
        plt.ylabel("Frequency")

        output_path = self.output_dir / f"{column_name}_histogram.png"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        return output_path

    def save_boxplot(self, series: pd.Series, column_name: str) -> Path:
        clean_series = series.dropna()

        plt.figure(figsize=(8, 6))
        plt.boxplot(clean_series, vert=True)
        plt.title(f"Boxplot of {column_name}")
        plt.ylabel(column_name)

        output_path = self.output_dir / f"{column_name}_boxplot.png"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        return output_path