''''
Esse sistema de pré-processamento automático tem como objetivo receber dados e limpar, padronizar
e preparar esse dados para treinamentos de modelos de Machine Learning

ideia: Ver funções no Pandas que me permitam fazer esses tratamentos automaticamente como os que eu anotei no caderno
provavelmente vou usar interpolação, mas tenho que ver como isso funciona pra string

esses dados serão exportados como tabela no final para que seja possível fazer a comparação
vou usar o read.csv 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #n sei se vou usar essa daqui
from sklearn.preprocessing import StandardScaler


file_path = 'datasets/missing_values_example.csv'

df = pd.read_csv(file_path) #lê o csv

#Verificando valores faltantes
#m_values = pd.isnull(df) 

#retirando as linhas que possuem valores nulos
df_sem_nulos = df.dropna()



#print(df.dtypes) #vê os tipos dos dados