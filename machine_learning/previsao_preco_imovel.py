import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  #regression metrics 
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt


file_path='./datasets/kc_house_data.csv'

df = pd.read_csv(file_path)

#print(df.head())
#print(df.columns)

#We have to do the data processing, removing some of the columns that are useless for the price prediction, such as id, date and zipcode
#That way, we can split the train dataset in features (x) and target, which is the price(y) 
#
#
#temos que fazer o processamento desses dados, retirando algumas columas que não interessam para a previsão de preços, como id, data e código postal
#assim podemos separar o dataset em treino em features (x) e target, que é o preço (y)

X = df.iloc[:, 3:].values
X = X[:, np.r_[0:13,14:18]]
y = df.iloc[:, 2].values

#Splitting train and test datasets

#separando treino e teste

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42 #usado para tornar o processo de distribuição dos dados de teste e treino, deterministico
                      #used to make the distribuituion os test and train data deterministic
)


#normalizando os dados para modelos que nessitam
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train) #APRENDE  e NORMALIZA os dados de treino
X_test_scaled = scaler.transform(X_test) #SOMENTE NORMALIZA os dados de teste, NÃO APRENDE COM ELES

#Criando os modelos de aprendizado supervisionado
linreg = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=25)
tree = DecisionTreeRegressor(random_state=42)
rtree = RandomForestRegressor(
    n_estimators=400,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features=1.0,
    random_state=42,
    n_jobs=-1
    )

#Aplicando os modelos (treino)
print("Treinando modelos...")

linreg.fit(X_train, y_train)

knn.fit(X_train_scaled, y_train)

tree.fit(X_train, y_train)

rtree.fit(X_train, y_train)


print("Modelos treinados!")

#Fazendo as previsões
print("Prevendo...")

pred_linreg = linreg.predict(X_test)
pred_knn = knn.predict(X_test)
pred_tree = tree.predict(X_test)
pred_rtree = rtree.predict(X_test)

print("Resultados prontos!")

#comparando os resultados
print("Linear regression")
print("MAE: ", mean_absolute_error(y_test, pred_linreg))
print("MSE: ", mean_squared_error(y_test, pred_linreg))
print("R2: ", r2_score(y_test, pred_linreg))

print("KNN")
print("MAE: ", mean_absolute_error(y_test, pred_knn))
print("MSE: ", mean_squared_error(y_test, pred_knn))
print("R2: ", r2_score(y_test, pred_knn))

print("Decision Tree")
print("MAE: ", mean_absolute_error(y_test, pred_tree))
print("MSE: ", mean_squared_error(y_test, pred_tree))
print("R2: ", r2_score(y_test, pred_tree))

print("Random Forest")
print("MAE: ", mean_absolute_error(y_test, pred_rtree))
print("MSE: ", mean_squared_error(y_test, pred_rtree))
print("R2: ", r2_score(y_test, pred_rtree))


#Plotando o gráfico