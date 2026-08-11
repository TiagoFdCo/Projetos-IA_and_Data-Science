"""
Este sistema de análse exploratória tem a função de ler um csv e, usando as bilbiotecas pandas (manipulação de dados)
e matplotlib (visualização dos dados), mostrar métricas e gráficos referentes ao dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

''''ps: eu tentei usar a contrabarra e deu erro na leitura do path pq o windows interpreta contrabarra U como 
'escape Unicode', como 'contrabarra n' ou 'contrabarra t'. A solução foi usar a barra normal, mas também poderia 
ser resolvida usando raw string (colocar r antes da aspas simples)'''

file_path = 'datasets/fuel_prices_1970_2026.csv'

df = pd.read_csv(file_path)

#print(df.head()) #head retorna as primeiras n linhas passadas como argumento. Por padrão, retorna as 5 primeiras

#Gerando estatísticas descritivas
print(round(df.describe(), 2)) #Mostra algumas estatísticas descritivas relativas ao dataset

#Criando o gráfico
ax = df.plot(x='Date', y='Crude_Oil_Price', title='Preço do petróleo com o passar dos anos')

#Customizando os eixos

ax.set_xlabel('Datas de referência')
ax.set_ylabel('Preço em dólar')

plt.show()

