![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

# Data Science & AI — Fundamentals

A collection of small Data Science and Machine Learning projects built from scratch to practice the full pipeline — from raw CSV handling to model comparison — using Python, pandas, NumPy, scikit-learn and matplotlib.

Each folder is a self-contained mini-project with increasing complexity, covering supervised classification, unsupervised clustering, regression, feature encoding and data preprocessing.

## Projects

| Project | Description | Task Type | Stack |
|---|---|---|---|
| [`/data_science`](./data_science) | Standalone data-handling utilities: BMI report reader, EDA system, preprocessing pipeline | Data Science fundamentals | pandas, matplotlib |
| [`/flower_species_prediction`](./flower_species_prediction) | Classifies Iris flowers into 3 species by comparing Logistic Regression, KNN and Naive Bayes | Supervised classification | scikit-learn |
| [`/flower_species_non_supervised_classification`](./flower_species_non_supervised_classification) | Groups Iris flowers into clusters with K-Means, using the elbow method to pick *k* | Unsupervised clustering | scikit-learn |
| [`/house_price_prediciton`](./house_price_prediciton) | Predicts house prices (King County dataset) comparing Linear Regression, KNN, Decision Tree and Random Forest | Regression | scikit-learn |
| [`/student_grade_prediction`](./student_grade_prediction) | Predicts students' final math grade from demographic and academic features, with and without prior grades as input | Regression | scikit-learn |
| [`/salary_prediction`](./salary_prediction) | 🚧 Work in progress | Regression | scikit-learn |

## Datasets

Datasets are not committed to this repository (see `.gitignore`) and must be downloaded separately into each project's `datasets/` folder:

* [Iris Species](https://www.kaggle.com/datasets/uciml/iris?select=Iris.csv)
* [KC House Data](https://www.kaggle.com/datasets/shivachandel/kc-house-data)
* [Math Students](https://www.kaggle.com/datasets/janiobachmann/math-students)
* [Global Fuel Price Trends 1970–2026](https://www.kaggle.com/datasets/minahilfatima12328/global-fuel-price-trends-19702026)
* [Jobs Dataset from Glassdoor](https://www.kaggle.com/datasets/thedevastator/jobs-dataset-from-glassdoor)

## Repository Structure

```
projetinho_ia/
├── data_science/
│   ├── csv_bmi_reader.py
│   ├── eda_system.py
│   ├── pre_prossecing_system.py
│   └── readme.md
├── flower_species_prediction/
│   ├── classificacao_flores_supervisionado.py
│   └── readme.md
├── flower_species_non_supervised_classification/
│   ├── classificacao_flores_com_clustering.py
│   └── readme.md
├── house_price_prediciton/
│   ├── previsao_preco_imovel.py
│   └── readme.md
├── student_grade_prediction/
│   ├── math_student_grade_prediction.py
│   └── readme.md
├── salary_prediction/
│   ├── salary_prediciton.py
│   └── readme.md
├── .gitignore
└── readme.md          <- this file
```

Each subfolder has its own `readme.md` with details on the dataset, approach and results for that specific project.

## License

This repository is licensed under the [MIT License](LICENSE).
