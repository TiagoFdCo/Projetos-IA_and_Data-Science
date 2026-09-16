'''
this dataset contains various features about students background, which may or may not impact on their grades,
especialy the final grade, which is the target we want to predict.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  #regression metrics 

'''
 To handle categorical variables in regression, we follow these steps:

1 - One-Hot Encoding: Convert categorical variables into binary columns, where each column corresponds to a unique category of the variable.
2 - Regression: Once the categorical variables are encoded, they can be used as features (independent variables) in a regression model.
3 - Fit a Linear Regression Model: Use the encoded features along with a target variable to fit a linear regression model.
'''

file_path = './datasets/student-mat.csv'

df = pd.read_csv(file_path)

'''
As for the labels we will order them the following way:

* numerical: age, absences, G1, G2

* binary: school, sex, address, famsize, Pstatus, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic

* Ordinal: Medu, Fedu, traveltime, studytime, failures, famrel, freetime, goout, dalc, walc, health

* Nominal: Mjob, Fjob, reason, guardian

* Target: G3
'''

#For the binary features
df['school'] = df['school'].map({
    'GP': 0,
    'MS': 1
})

df['sex'] = df['sex'].map({
    'F': 0,
    'M': 1
})

df['address'] = df['address'].map({
    'U': 0,
    'R': 1
})

df['famsize'] = df['famsize'].map({
    'GT3': 0,
    'LE3': 1
})

df['Pstatus'] = df['Pstatus'].map({
    'A': 0,
    'T': 1
})

#for yes/no features:
binary_cols = [
    'schoolsup',
    'famsup',
    'paid',
    'activities',
    'nursery',
    'higher',
    'internet',
    'romantic'
]

for col in binary_cols:
    df[col] = df[col].map({
        'no': 0,
        'yes': 1
    })


#Ordinal features:
ordinal_cols = [
    'Medu',
    'Fedu',
    'traveltime',
    'studytime',
    'failures',
    'famrel',
    'freetime',
    'goout',
    'Dalc',
    'Walc',
    'health'
]

#Nominal features:
nominal_cols = [
    'Mjob',
    'Fjob',
    'reason',
    'guardian'
]

#Numerical features:
numeric_cols = [
    'age',
    'absences',
    'G1',
    'G2'
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            'nominal',
            OneHotEncoder(handle_unknown='ignore'), 
            nominal_cols
         ),
         (
             'numeric',
             StandardScaler(),
             numeric_cols
         )
    ],
    remainder='passthrough'
)

#Splitting the dataset in training set and test set
X = df.drop('G3', axis=1)
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

#creating and fitting the sets in the model

linreg = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=20)
tree = DecisionTreeRegressor(random_state=42)

#Applying the model
print("Treinando modelo ...")

linreg.fit(X_train, y_train)

knn.fit(X_train, y_train)

tree.fit(X_train, y_train)

print("Modelos treinados!")

#predicting
print("\n Prevendo...")
pred_linreg = linreg.predict(X_test)
pred_knn = knn.predict(X_test)
pred_tree = tree.predict(X_test)

print("Resultados prontos!")

#comparando os resultados
print("Linear regression")
print("MAE: ", mean_absolute_error(y_test, pred_linreg))
print("MSE: ", mean_squared_error(y_test, pred_linreg))
print("R2: ", r2_score(y_test, pred_linreg))

print("Decision Tree")
print("MAE: ", mean_absolute_error(y_test, pred_tree))
print("MSE: ", mean_squared_error(y_test, pred_tree))
print("R2: ", r2_score(y_test, pred_tree))

print("KNN")
print("MAE: ", mean_absolute_error(y_test, pred_knn))
print("MSE: ", mean_squared_error(y_test, pred_knn))
print("R2: ", r2_score(y_test, pred_knn))


'''
Testing now without G1 and G2
'''

preprocessor2 = ColumnTransformer(
    transformers=[
        (
            'nominal',
            OneHotEncoder(handle_unknown='ignore'),
            nominal_cols
        ),
        (
            'numeric',
            StandardScaler(),
            ['age', 'absences']
        )
    ],
    remainder='passthrough'
)

X2 = df.drop(['G3', 'G2', 'G1'], axis=1)
y2 = df['G3']

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2,
    y2,
    test_size=0.2,
    random_state=42
)

X2_train = preprocessor2.fit_transform(X2_train)
X2_test = preprocessor2.transform(X2_test)

#creating and fitting the sets in the model

linreg = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=20)
tree = DecisionTreeRegressor(random_state=42)

#Applying the model
print("Treinando modelo ...")

linreg.fit(X2_train, y2_train)

knn.fit(X2_train, y2_train)

tree.fit(X2_train, y2_train)

print("Modelos treinados!")

#predicting
print("\n Prevendo...")
pred_linreg = linreg.predict(X2_test)
pred_knn = knn.predict(X2_test)
pred_tree = tree.predict(X2_test)

print("Resultados prontos!")

#comparando os resultados
print("Linear regression")
print("MAE: ", mean_absolute_error(y2_test, pred_linreg))
print("MSE: ", mean_squared_error(y2_test, pred_linreg))
print("R2: ", r2_score(y2_test, pred_linreg))

print("Decision Tree")
print("MAE: ", mean_absolute_error(y2_test, pred_tree))
print("MSE: ", mean_squared_error(y2_test, pred_tree))
print("R2: ", r2_score(y2_test, pred_tree))

print("KNN")
print("MAE: ", mean_absolute_error(y2_test, pred_knn))
print("MSE: ", mean_squared_error(y2_test, pred_knn))
print("R2: ", r2_score(y2_test, pred_knn))