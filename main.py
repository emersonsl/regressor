import math
from random import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

#DATASET

#criando data frame para armazenar os dados do dataset
df = pd.DataFrame(columns=['X','Y'])

#arbitrando valor de a
a  = 250

#calculando valor de y; adicionando valores de X e Y ao dataset
for x in range(0,1000,1):
    y = math.sqrt((2*math.sqrt((a**3)*(a+2*x)))+(2*(a**2))+(2*a*x)-(x**2))
    df = df.append({'X':x, 'Y':y}, ignore_index=True)

#exportando arquivo com dataset
df.to_csv('dataset.csv', index=False)

print("DATA SET\n")
print(df)
print("tamanho do DataSet: "+str(df.shape))

#REDE NEURAL

#separado dados de alvo dos dados entrada

inp = df['X']

out = df['Y']

#separando conjunto de treino do conjunto de teste, proporção 70, 30, respectivamente

inp_train, inp_test, out_train, out_test = train_test_split(inp.values.reshape(-1,1), out, test_size=0.3)

print('tamanho do conjunto entrada de treino: '+str(inp_train.shape))
print('tamanho do conjunto saída de treino: '+str(out_train.shape))

print('tamanho do conjunto entrada de test: '+str(inp_test.shape))
print('tamanho do conjunto saída de test: '+str(out_test.shape)+'\n')


#treinando a rede

regr = MLPRegressor(max_iter=1500, activation='relu', hidden_layer_sizes=350)
regr.fit(inp_train, out_train)

#testando rede
print("acuracia: "+str(regr.score(inp_test, out_test)))