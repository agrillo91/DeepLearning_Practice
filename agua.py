# -*- coding: utf-8 -*-

#Importar librerias básicas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Iportar Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Importar datos
datos = pd.read_csv("potabilidad_agua.csv")

#Separar datos
x = datos.drop('Potabilidad', axis=1)
y = datos['Potabilidad'] 

#Visualizar cantidad en cada clase
y.value_counts()

# Datos Nan
''' METODO 1 COLUMNA POR COLUMNA
x.isna().sum()
x.ph.fillna(value=x.ph.mean(), inplace=True)
x.isna().sum()
print(x.isna().sum())
'''

''' METODO 2, TODAS LAS COLUMNAS '''
# Datos Nan
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x[:] = imputer.fit_transform(x) 

 #Normalizacion de datos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

#Separar Datos de Entrenamiento y Pruebas
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

'''RED NEURONAL'''
#Crear modelo de Red Neuronal
red = Sequential()
# Crear capas
entrada = Dense(units=9, activation='relu')
oculta1 = Dense(units=500, activation='relu')
oculta2 = Dense(units=500, activation='relu')
salida = Dense(units=1, activation='sigmoid')

#Incorporar Capas a Red
red.add(entrada)
red.add(oculta1)
red.add(oculta2)
red.add(salida)

#Compilacion de Red
red.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01),
    loss = 'binary_crossentropy',
    metrics =['accuracy'])

 #1er entrenamiento 
history = red.fit(xtrain, ytrain, batch_size=30, epochs=100)
# 2do Entrenamiento
history2 = red.fit(xtrain, ytrain, batch_size=50, epochs=100) 

'''EVALUACIÓN'''
plt.plot(history.history['accuracy'] + history2.history['accuracy'])
plt.show()

plt.plot(history.history['loss'] + history2.history['loss'])
plt.show()

#Valoracion
y_prediction= red.predict(xtest)
#Redondeo para 0 y 1
y_prediction = np.round(y_prediction)

from sklearn.metrics import accuracy_score, confusion_matrix
score = accuracy_score(ytest, y_prediction)
result = red.evaluate(xtest, ytest)

matrix = confusion_matrix(ytest, y_prediction)













