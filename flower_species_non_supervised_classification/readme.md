[⬅ Back to portfolio root](../readme.md)

# Flower Species Clustering (Unsupervised)

Groups Iris flowers into clusters using K-Means, without using the species labels — the number of clusters is chosen with the elbow method instead of being told in advance.

## Overview

This is the unsupervised counterpart to [`/flower_species_prediction`](../flower_species_prediction): instead of learning from labeled species, K-Means groups flowers purely by the similarity of their measurements. The interesting part is that the elbow method independently converges on *k = 3* — the same number of species that actually exist in the dataset.

## Dataset

* **Source**: [Iris dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/iris?select=Iris.csv)
* **Size**: 150 rows, 6 columns
* **Features used**:
  * `SepalLengthCm` (numerical)
  * `SepalWidthCm` (numerical)
  * `PetalLengthCm` (numerical)
  * `PetalWidthCm` (numerical)

The `Id` and `Species` columns are both dropped — `Species` can't be used since this is unsupervised learning. Features are standardized before clustering.

## Approach

* **K-Means**: partitions the data into *k* clusters by minimizing the distance between each point and its cluster's centroid.
* **Elbow method**: K-Means is run for *k* = 1 to 5, tracking each run's *inertia* (sum of squared distances to the nearest centroid). The "elbow" in the resulting plot — where adding another cluster stops meaningfully reducing inertia — indicates a good value of *k*.

Once *k* is chosen, the final model is trained with `n_init=10` (10 different centroid initializations, keeping the best) and the resulting clusters are plotted using petal length and petal width.

## Results

*(Fill in after running `python classificacao_flores_com_clustering.py`)*

* **Elbow plot**: *(describe where the elbow appears — expected around k=3)*
* **Cluster plot**: *(describe how well-separated the 3 clusters look on petal length × petal width)*


**Conclusion**: *(e.g. "K-Means independently found 3 natural groupings that closely match the real species boundaries, confirming petal measurements alone are strong separators for this dataset.")*

## Project Structure

```
flower_species_non_supervised_classification/
├── datasets/
│   └── Iris.csv
├── assets/
│   ├── elbow_method.png
│   └── clusters.png
├── classificacao_flores_com_clustering.py
└── readme.md
```

## Installation

```bash
git clone https://github.com/[seu-usuario]/projetinho_ia.git
cd projetinho_ia/flower_species_non_supervised_classification
pip install pandas numpy scikit-learn matplotlib
```

Download [Iris.csv](https://www.kaggle.com/datasets/uciml/iris?select=Iris.csv) into a local `datasets/` folder inside this directory.

## Usage

```bash
python classificacao_flores_com_clustering.py
```

This opens the elbow-method plot first, then trains the final K-Means model with `k=3` and opens the resulting cluster scatter plot.

## Tech Stack

* Python 3.11
* pandas / numpy
* scikit-learn
* matplotlib

## Possible Improvements

* Evaluate clustering quality with silhouette score, not just inertia
* Compare cluster assignments against the true `Species` labels (for validation only, never for training)
* Try other clustering algorithms (e.g. DBSCAN, Agglomerative Clustering) for comparison
