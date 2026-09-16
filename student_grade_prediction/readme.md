[⬅ Back to portfolio root](../readme.md)

# Student Grade Prediction

Predicts a student's final math grade from demographic, social and academic features, comparing three regression models — and testing how much the prediction relies on already knowing the student's earlier grades.

## Overview

This project explores a common trap in grade-prediction problems: if you include a student's first and second period grades (`G1`, `G2`) as features, predicting the final grade (`G3`) becomes almost trivial, since grades are highly correlated across periods. To make the comparison meaningful, the models are trained **twice** — once with `G1`/`G2` included, and once without them — to see how much predictive power comes from genuinely independent features (school, family background, study habits, etc.) versus just extrapolating from prior grades.

## Dataset

* **Source**: [Math Students (Kaggle)](https://www.kaggle.com/datasets/janiobachmann/math-students)
* **Features**, grouped by type:
  * **Numerical**: `age`, `absences`, `G1`, `G2`
  * **Binary**: `school`, `sex`, `address`, `famsize`, `Pstatus`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`
  * **Ordinal**: `Medu`, `Fedu`, `traveltime`, `studytime`, `failures`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`
  * **Nominal**: `Mjob`, `Fjob`, `reason`, `guardian`
* **Target**: `G3` (final grade)

Binary categorical columns are manually mapped to 0/1. Nominal columns are one-hot encoded via a `ColumnTransformer`, and numerical columns are standardized. Ordinal columns are passed through as-is, since they're already encoded as small integers in the source data.

## Approach

Two experiments are run with the same three models:

* **Linear Regression**
* **K-Nearest Neighbors** (`n_neighbors=20`)
* **Decision Tree**

**Experiment 1**: all features included (`G1`, `G2` + demographic/social features).
**Experiment 2**: `G1` and `G2` removed, keeping only demographic/social/behavioral features.

## Metrics

* **MAE**: average absolute error, in grade points.
* **MSE**: average squared error — penalizes large misses more.
* **R²**: proportion of grade variance explained by the model.

## Results

**With G1/G2 (Experiment 1)**

| Model              | MAE | MSE | R²   |
|--------------------|-----|-----|------|
| Linear Regression  | —   | —   | —    |
| Decision Tree       | —   | —   | —    |
| KNN                 | —   | —   | —    |

**Without G1/G2 (Experiment 2)**

| Model              | MAE | MSE | R²   |
|--------------------|-----|-----|------|
| Linear Regression  | —   | —   | —    |
| Decision Tree       | —   | —   | —    |
| KNN                 | —   | —   | —    |

*(Fill in after running `python math_student_grade_prediction.py` — both experiments print to the terminal in sequence.)*

**Conclusion**: *(Compare the two tables — expect R² to drop substantially in Experiment 2, showing how much the models were relying on G1/G2 rather than genuinely independent features.)*

## Project Structure

```
student_grade_prediction/
├── datasets/
│   └── student-mat.csv
├── math_student_grade_prediction.py
└── readme.md
```

## Installation

```bash
git clone https://github.com/[seu-usuario]/projetinho_ia.git
cd projetinho_ia/student_grade_prediction
pip install pandas numpy scikit-learn matplotlib
```

Download [student-mat.csv](https://www.kaggle.com/datasets/janiobachmann/math-students) into a local `datasets/` folder inside this directory.

## Usage

```bash
python math_student_grade_prediction.py
```

This runs both experiments in sequence and prints MAE, MSE and R² for each model in each experiment.

## Tech Stack

* Python 3.11
* pandas / numpy
* scikit-learn
* matplotlib

## Possible Improvements

* Cross-validation instead of a single train/test split
* Feature importance analysis (e.g. via the Decision Tree or a Random Forest) to identify which non-grade features matter most
* Hyperparameter tuning for KNN and Decision Tree
