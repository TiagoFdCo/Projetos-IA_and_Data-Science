import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

file_path = './datasets/Iris.csv'

df = pd.read_csv(file_path)


#print(df.head())


#checking for null values
#print(pd.isnull(df))


#removing id (which we don't need) and species column (which we can't have in features)
X = df.iloc[:, 1:5].values  #'take all the values from the rows, from column 1 to before fifth'

#labeling the target column because they are strings and the other parameters are numbers
encoder = LabelEncoder()

y = encoder.fit_transform(df.iloc[:, 5])    #take only the species (target)

#splitting into train and test

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#standartizing the data for the models who need it
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train) #APRENDE  e NORMALIZA os dados de treino
X_test_scaled = scaler.transform(X_test) #SOMENTE NORMALIZA os dados de teste, NÃO APRENDE COM ELES

#Creating the supervised learning models
logreg = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=20)
bayes = GaussianNB()


#training the models
logreg.fit(X_train_scaled, y_train)

knn.fit(X_train_scaled, y_train)

bayes.fit(X_train_scaled, y_train)

#starting the prediction
pred_logreg = logreg.predict(X_test_scaled)

pred_knn= knn.predict(X_test_scaled)

pred_bayes = bayes.predict(X_test_scaled)


#printing metrics
print("Logistic regression")
print("Accuracy: ", accuracy_score(y_test, pred_logreg))
print("Precision: ", precision_score(y_test, pred_logreg, average="weighted"))
print("Recall: ", recall_score(y_test, pred_logreg, average="weighted"))
print("F1: ", f1_score(y_test, pred_logreg, average="weighted"))

print("\nKNN")
print("Accuracy: ", accuracy_score(y_test, pred_knn))
print("Precision: ", precision_score(y_test, pred_knn, average="weighted"))
print("Recall: ", recall_score(y_test, pred_knn, average="weighted"))
print("F1: ", f1_score(y_test, pred_knn, average="weighted"))

print("\nNaive Bayes")
print("Accuracy: ", accuracy_score(y_test, pred_bayes))
print("Precision: ", precision_score(y_test, pred_bayes, average="weighted"))
print("Recall: ", recall_score(y_test, pred_bayes, average="weighted"))
print("F1: ", f1_score(y_test, pred_bayes, average="weighted"))

#plotting confusion matrices
ConfusionMatrixDisplay.from_predictions(y_test, pred_logreg)
plt.title("Logistic Regression")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, pred_knn)
plt.title("KNN")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, pred_bayes)
plt.title("Naive Bayes")
plt.show()



