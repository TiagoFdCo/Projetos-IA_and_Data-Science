[⬅ Back to portfolio root](../readme.md)

# House Price Prediction

Predicts house sale prices from property features (King County, USA dataset), comparing four regression models: Linear Regression, KNN, Decision Tree and Random Forest.

## Overview

This is a regression problem — instead of predicting a category, the goal is to predict a continuous value (price) from a property's characteristics such as bedrooms, bathrooms, square footage and condition. Four models of increasing complexity are trained and compared to see how much predictive power is gained by moving from a simple linear model to an ensemble method.

## Dataset

* **Source**: [KC House Data (Kaggle)](https://www.kaggle.com/datasets/shivachandel/kc-house-data)
* **Features used**: property attributes such as bedrooms, bathrooms, square footage, floors, waterfront, view, condition, grade, year built/renovated, and location fields (columns 3 through 18, excluding `zipcode`)
* **Target**: `price`

The `id`, `date` and `zipcode` columns are dropped — `id` and `date` carry no predictive signal for this approach, and `zipcode` is excluded here (see *Possible Improvements* below for how it could be used instead of discarded). Features are standardized for KNN; the tree-based models use the raw (unscaled) features since they don't require it.

## Approach

* **Linear Regression**: fits a straight-line relationship between features and price.
* **K-Nearest Neighbors**: predicts a price by averaging the prices of the most similar properties.
* **Decision Tree**: splits the data into progressively purer subsets based on feature thresholds.
* **Random Forest**: an ensemble of 400 decision trees (`max_depth=20`, `min_samples_split=5`, `min_samples_leaf=2`), averaging their predictions to reduce overfitting.

## Metrics

* **MAE (Mean Absolute Error)**: average absolute difference between predicted and actual price, in dollars.
* **MSE (Mean Squared Error)**: average squared difference — penalizes large errors more heavily than MAE.
* **R² Score**: proportion of price variance explained by the model (closer to 1 is better).

## Results

| Model              | MAE | MSE | R²   |
|--------------------|-----|-----|------|
| Linear Regression  | —   | —   | —    |
| KNN                 | —   | —   | —    |
| Decision Tree       | —   | —   | —    |
| Random Forest       | —   | —   | —    |

*(Fill in after running `python previsao_preco_imovel.py`)*

**Conclusion**: *(Which model generalized best and why — e.g. Random Forest usually outperforms a single Decision Tree by reducing variance, while Linear Regression may struggle if price has non-linear relationships with the features.)*

## Project Structure

```
house_price_prediciton/
├── datasets/
│   └── kc_house_data.csv
├── previsao_preco_imovel.py
└── readme.md
```

## Installation

```bash
git clone https://github.com/[seu-usuario]/projetinho_ia.git
cd projetinho_ia/house_price_prediciton
pip install pandas numpy scikit-learn matplotlib
```

Download [kc_house_data.csv](https://www.kaggle.com/datasets/shivachandel/kc-house-data) into a local `datasets/` folder inside this directory.

## Usage

```bash
python previsao_preco_imovel.py
```

This prints MAE, MSE and R² for all four models to the terminal.

## Tech Stack

* Python 3.11
* pandas / numpy
* scikit-learn
* matplotlib

## Possible Improvements

* Encode `zipcode` (e.g. one-hot or target encoding) instead of discarding it — location is usually one of the strongest price predictors
* Hyperparameter tuning for Random Forest and KNN via `GridSearchCV`
* Log-transform `price` before training, since housing prices are typically right-skewed
* Add a results plot (predicted vs. actual price) — the script currently has an empty "Plotando o gráfico" section at the end
