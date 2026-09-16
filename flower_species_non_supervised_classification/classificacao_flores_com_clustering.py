#Non-supervised learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans #create a object with the KMeans model, while k_means is in the function format, with no object being created
from sklearn.preprocessing import StandardScaler


file_path = './datasets/Iris.csv'

df = pd.read_csv(file_path)

#removing id (which we don't need) and species column (which we can't have in unsupervized learning)
X = df.iloc[:, 1:5].values

#standartizng the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#figuring how many clusters are needed for the sample using the elbow method
inertias = []

for k in range(1, 6): #will ask the question: "what if I use k = 1 cluster? what about k = 2 clusters? etc."
    k_means = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    k_means.fit(X_scaled)

    inertias.append(k_means.inertia_) 

plt.plot(range(1, 6), inertias, marker='o')
plt.grid
plt.show() 

#we see from this method, that k=3 is a good number, which is true, because in the original dataset, we have 3 species
#now we train the model

k_means = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

k_means.fit(X_scaled)
print(k_means.labels_)

#plotting the clusters in a graph
plt.figure(figsize=(10, 6))

plt.scatter(
    X_scaled[:, 2], 
    X_scaled[:, 3],
    c=k_means.labels_
)

plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.title("Clusters encontrados pelo K-Means")
plt.show()