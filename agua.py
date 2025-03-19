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

from keras import regularizers
from tensorflow.keras.layers import Dropout

'''RED NEURONAL'''
#Crear modelo de Red Neuronal
red = Sequential()
# Crear capas
entrada = Dense(units=5, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.001, l2=0.001))
oculta1 = Dense(units=5, activation='relu', kernel_regularizer = regularizers.l1_l2(l1=0.001, l2=0.001))
salida = Dense(units=1, activation='sigmoid')

#Incorporar Capas a Red
red.add(entrada)
red.add(Dropout(0.5))
red.add(oculta1)
red.add(Dropout(0.5))
red.add(salida)

#Compilacion de Red
red.compile(
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01),
    loss = 'binary_crossentropy',
    metrics =['accuracy'])

x_train_new = xtrain[1620:]
x_val = xtrain[:1000]

y_train_new = ytrain[1620:]
y_val = ytrain[:1000]

'''
 #1er entrenamiento  
history = red.fit(xtrain, ytrain, batch_size=30, epochs=100) 

# 2do Entrenamiento
history2 = red.fit(xtrain, ytrain, batch_size=50, epochs=100) 
'''

# 3er Entrenamiento
history = red.fit(x_train_new, y_train_new,
                  validation_data = (x_val, y_val), 
                  batch_size=100, 
                  epochs=15)

loss = history.history['loss']
val_loss = history.history['val_loss']
epocas = range(1, len(loss) + 1)

plt.plot(epocas, loss, 'bo', label='Perdida de entrenamiento')
plt.plot(epocas, val_loss, 'b', label='Perdida de Evaluacion')
plt.title('Gráfica de Perdidas')
plt.legend()
plt.xlabel('Epocas')
plt.ylabel('Pérdida')
plt.show()

precision = history.history['accuracy']
val_precision = history.history['val_accuracy']
epocas = range(1, len(precision) + 1)

plt.plot(epocas, precision, 'bo', label='Precisión de entrenamiento')
plt.plot(epocas, val_precision, 'b', label='Precisión de Evaluacion')
plt.title('Gráfica de Precision')
plt.legend()
plt.xlabel('Epocas')
plt.ylabel('Precision')
plt.show()

'''
plt.plot(history.history['accuracy'] + history2.history['accuracy'])
plt.show()

plt.plot(history.history['loss'] + history2.history['loss'])
plt.show()
''' 

#Valoracion
y_prediction= red.predict(xtest)
#Redondeo para 0 y 1
y_prediction = np.round(y_prediction)

from sklearn.metrics import accuracy_score, confusion_matrix
score = accuracy_score(ytest, y_prediction)
result = red.evaluate(xtest, ytest)

matrix = confusion_matrix(ytest, y_prediction)













