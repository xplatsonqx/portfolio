# Automated Exploratory Data Analysis Tool

## 🔍 Overview

This project is an object-oriented Python tool designed to automate exploratory data analysis (EDA) for structured CSV datasets.

It loads data, classifies columns, performs statistical analysis, and generates visualizations and summary reports.

---

## 🚀 Features

* automatic dataset loading
* numeric and categorical column detection
* statistical summary (mean, std, quartiles)
* missing values analysis
* histogram and boxplot generation
* exportable CSV report

---

## 🧠 Tech stack

* Python
* pandas
* matplotlib

---

## 📊 Example output

### Summary report

The tool generates a CSV file containing descriptive statistics for numeric columns.

### Visualizations

* histograms
* boxplots

(see `/outputs/plots/`)

---

## ▶️ How to run

```bash
pip install -r requirements.txt
python src/main.py
```

---

## 📁 Project structure

```
portfolio/
├── src/
├── data/
├── outputs/
└── README.md
```

---

## 💡 Motivation

Exploratory data analysis is often repetitive.
This project demonstrates how to automate core EDA tasks using an object-oriented approach.

---

## 📌 Future improvements

* support for categorical analysis
* correlation heatmaps
* automated report generation (PDF)
