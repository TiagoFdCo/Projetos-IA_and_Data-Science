[⬅ Back to portfolio root](../readme.md)

# Data Science Utilities

A small collection of standalone scripts practicing core Data Science skills — reading and reporting on raw data, exploratory data analysis (EDA), and preprocessing — before moving into full ML pipelines.

## Overview

Unlike the other folders in this repository, this one isn't a single end-to-end project. It's a set of independent scripts, each focused on one fundamental skill:

| Script | Skill practiced | What it does |
|---|---|---|
| `csv_bmi_reader.py` | Manual data parsing & reporting | Reads a CSV of height/weight from stdin and prints a BMI report (averages, min/max) — no pandas, pure Python |
| `eda_system.py` | Exploratory Data Analysis | Loads a CSV with pandas, prints descriptive statistics, and plots a trend over time with matplotlib |
| `pre_prossecing_system.py` | Data preprocessing | Loads a CSV and removes rows with missing values as a first preprocessing step |

## Scripts

### `csv_bmi_reader.py`
Takes CSV input line-by-line from the terminal (`ID, height, weight` format) and computes: average height, average weight, average BMI, and the min/max height and weight. Written without pandas, to practice raw data handling.

### `eda_system.py`
Reads a fuel-price time series ([Global Fuel Price Trends dataset](https://www.kaggle.com/datasets/minahilfatima12328/global-fuel-price-trends-19702026)) and generates descriptive statistics (`df.describe()`) plus a line plot of crude oil price over time.

### `pre_prossecing_system.py`
Reads a CSV with missing values and drops rows containing nulls, as a first step toward a proper preprocessing pipeline. *(Work in progress — see Possible Improvements.)*

## Installation

```bash
git clone https://github.com/[seu-usuario]/projetinho_ia.git
cd projetinho_ia/data_science
pip install pandas numpy matplotlib scikit-learn
```

Each script expects its own CSV inside a local `datasets/` folder:
* `eda_system.py` → `datasets/fuel_prices_1970_2026.csv` ([source](https://www.kaggle.com/datasets/minahilfatima12328/global-fuel-price-trends-19702026))
* `pre_prossecing_system.py` → `datasets/missing_values_example.csv`
* `csv_bmi_reader.py` reads directly from stdin (no file needed)

## Usage

```bash
python csv_bmi_reader.py            # then paste CSV lines, empty line to finish
python eda_system.py
python pre_prossecing_system.py
```

## Tech Stack

* Python 3.11
* pandas / numpy
* matplotlib
* scikit-learn (`StandardScaler`, for future preprocessing steps)

## Possible Improvements

* Extend `pre_prossecing_system.py` to handle missing values via imputation (mean/median/interpolation) instead of only dropping rows, and to export the cleaned dataset for comparison
* Add outlier detection to the EDA system
* Turn `csv_bmi_reader.py` into a pandas-based version for comparison against the manual implementation
