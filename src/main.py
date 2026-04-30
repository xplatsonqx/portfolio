# Importing libraries
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
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

def show_data(df,column):
    print(f"\n{num_col[column]}:")
    print("Ile rekordów: ",series.count(),"\n")
    print("Ile unikalnych:", series.nunique(), "\n")
    print("Min:",df.min(),"Max:",df.max(),"\n")
    print("Średnia:",df.mean(), "Mediana:", df.median())

# =========================
# SETTINGS
# =========================

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)

mode = [3]
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


if 2 in mode:
    show_data(series, 0)

if 3 in mode:
    series = pd.to_numeric(df[num_col[0]], errors='coerce')
    counts = series.value_counts()
    print(counts.head())


    # plt.figure(figsize=(10, 6))
    # sns.barplot(x=counts.index, y=counts.values)
    #
    # plt.xlabel("Wartość")
    # plt.ylabel("Liczba wystąpień")
    # plt.title("Częstość występowania wartości")
    #
    # plt.xticks(rotation=90)  # ważne przy wielu wartościach
    # plt.tight_layout()
    # plt.show()


# =========================
# NUMERICAL COLUMNS
# 1. Patient name
# 1.1 Histogram – kluczowe obserwacje:

# - Rozkład płaski (brak piku)
# - wartości występują z podobną częstością
# - Brak koncentracji danych
# - nie ma zakresów, gdzie dane się „zbierają”
# - pełne pokrycie zakresu (~0–10 000)
# - brak luk, dane równomiernie rozłożone
# - Brak skośności (symetria)
# - brak ogonów w lewo/prawo
# - Brak widocznych anomalii
# - żadnych ekstremalnych koszy
# - sugestia rozkładu jednostajnego (uniform)
# - dane przypominają losowe rozłożenie

#  1.2 Boxplot – kluczowe obserwacje
#  - Mediana ≈ środek zakresu (~5000)
#  - silna symetria danych
#  - Q1 ≈ 2500, Q3 ≈ 7500
#  - środkowe 50% danych zajmuje dużą część zakresu
#  - Duży i równomierny IQR
#  - brak skupienia wokół jednej wartości
#  - Symetryczne wąsy (min–max)
#  - brak przesunięcia rozkładu
#  - Brak outlierów
#  - żadnych obserwacji odstających
#
#
#  1.3 Wnioski analityczne
#  - dane losowe / syntetyczne
#  - identyfikator (ID, indeks)
#  - zmienną o niskiej wartości informacyjnej
#  - „Zbyt idealny” rozkład to potencjalna czerwona flaga w danych rzeczywistych
#  - Prawdopodbnie jest to przypisany nr do pacjenta i uzywany jest zamiast jego imienia i nazwiska ze wzgledu bezpieczemstwa.
#W
# =========================
