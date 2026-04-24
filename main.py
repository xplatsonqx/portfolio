# Importing libraries
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("MacOSX")


# =========================
# FUNCTIONS
# =========================

def drawing_histogram(series, name):
    series = series.dropna()

    plt.figure()
    plt.hist(series, bins=30, edgecolor='black')
    plt.title(f"Histogram dla {name}")
    plt.xlabel(name)
    plt.ylabel("Liczba obserwacji")



def drawing_boxplot(series, name):
    series = series.dropna()

    plt.figure()
    plt.boxplot(series, vert=False)
    plt.title(f"Boxplot dla {name}")
    plt.xlabel(name)


def drawing_histogram_kde(series, name):
    series = series.dropna()

    plt.figure()
    plt.hist(series, bins=30, density=True, alpha=0.5, edgecolor='black')
    series.plot(kind='kde')

    plt.title(f"Histogram + KDE dla {name}")
    plt.xlabel(name)
    plt.ylabel("Gęstość")


# =========================
# SETTINGS
# =========================

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)

mode = [1]
##[0] column names (classified as NUMERIC, CATEGORICAL and DATE for further analyse)
##[1] numeric columns analysis
##[2]
# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "/Users/mateuszplata/Desktop/Portfolio/DrugSalesData.csv",
    sep=";",
    low_memory=False
)
# =========================
# COLUMN CLASSIFICATION
# =========================

cat_col = []
num_col = []
date_col = []

for col in df.columns:

    if "date" in col.lower() or "time" in col.lower():
        date_col.append(col)

    elif pd.api.types.is_numeric_dtype(df[col]):
        num_col.append(col)

    else:
        cat_col.append(col)

# =========================
# DEBUG INFO
# =========================

if 0 in mode:
    print("\nNUMERICAL columns:")
    print(num_col)

    print("\nCATEGORICAL columns:")
    print(cat_col)

    print("\nDATE columns:")
    print(date_col)

# =========================
# ANALYSIS
# =========================
if 1 in mode:
    for col in num_col:
        series = pd.to_numeric(df[col], errors='coerce').dropna()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # HISTOGRAM
        if len(series) >= 1:
            axes[0].hist(series, bins=30, edgecolor='black')
            axes[0].set_title(f"{col} - Histogram")
        else:
            axes[0].text(0.5, 0.5, "Brak danych", ha='center')

        # BOXPLOT
        if len(series) >= 1:
            axes[1].boxplot(series, vert=False)
            axes[1].set_title(f"{col} - Boxplot")
        else:
            axes[1].text(0.5, 0.5, "Brak danych", ha='center')

        # KDE
        if len(series) >= 3 and series.nunique() > 1:
            series.plot(kind='kde', ax=axes[2])
            axes[2].set_title(f"{col} - KDE")
        else:
            axes[2].text(0.5, 0.5, "Brak danych do KDE", ha='center')

        plt.tight_layout()
        plt.show()
