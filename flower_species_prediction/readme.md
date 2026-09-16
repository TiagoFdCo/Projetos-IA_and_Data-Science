[⬅ Back to portfolio root](../readme.md)

# Flower Species Classification (Supervised)

Classifies Iris flowers into three species by comparing three supervised ML models — Logistic Regression, K-Nearest Neighbors and Naive Bayes — based on sepal and petal measurements.

## Overview

This is a classic multi-class classification problem: given a flower's measurements, predict which of three species it belongs to. Rather than picking one model, this project trains and compares three different classifiers to see how they perform on the same data.

## Dataset

* **Source**: [Iris dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/iris?select=Iris.csv)
* **Size**: 150 rows, 6 columns
* **Features**:
  * `SepalLengthCm` (numerical)
  * `SepalWidthCm` (numerical)
  * `PetalLengthCm` (numerical)
  * `PetalWidthCm` (numerical)
* **Target**: `Species` — 3 classes: *Iris-setosa*, *Iris-versicolor*, *Iris-virginica*

The `Id` column is dropped since it carries no predictive information. The target is label-encoded (it's categorical text) and features are standardized before training.

## Approach

* **Logistic Regression**: a linear model combined with a sigmoid function to convert the result into a probability.
* **K-Nearest Neighbors**: classifies a sample based on the majority class among its nearest neighbors.
* **Naive Bayes**: classifies based on conditional probability, assuming feature independence.

All three are trained on the same 80/20 train/test split and standardized features, so the comparison is apples-to-apples.

## Metrics

* **Accuracy**: how many predictions were correct overall.
* **Precision**: of everything predicted as a given class, how much was actually that class.
* **Recall**: of everything that truly belongs to a class, how much was correctly found.
* **F1 Score**: harmonic mean of Precision and Recall — useful when classes are imbalanced.

Metrics are printed to the terminal, and a confusion matrix is plotted for each model.

## Results

| Model               | Accuracy | Precision | Recall | F1   |
|----------------------|----------|-----------|--------|------|
| Logistic Regression  | —        | —         | —      | —    |
| KNN                   | —        | —         | —      | —    |
| Naive Bayes           | —        | —         | —      | —    |

*(Fill in after running `python classificacao_flores_supervisionado.py` — the terminal output prints all four metrics per model.)*

**Conclusion**: *(Which model performed best, and why — e.g. the Iris classes are known to be almost linearly separable, so linear/probabilistic models tend to do very well here.)*

## Project Structure

```
flower_species_prediction/
├── datasets/
│   └── Iris.csv
├── classificacao_flores_supervisionado.py
└── readme.md
```

## Installation

```bash
git clone https://github.com/[seu-usuario]/projetinho_ia.git
cd projetinho_ia/flower_species_prediction
pip install pandas numpy scikit-learn matplotlib
```

Download [Iris.csv](https://www.kaggle.com/datasets/uciml/iris?select=Iris.csv) into a local `datasets/` folder inside this directory.

## Usage

```bash
python classificacao_flores_supervisionado.py
```

This prints the four metrics for each of the three models and opens a confusion matrix plot for each one.

## Tech Stack

* Python 3.11
* pandas / numpy
* scikit-learn
* matplotlib

## Possible Improvements

* Hyperparameter tuning for KNN (`n_neighbors` is currently fixed at 20) via `GridSearchCV`
* Cross-validation instead of a single train/test split
* Feature importance / decision boundary visualization
