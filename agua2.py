from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd

# Importar datos
datos = pd.read_csv("potabilidad_agua.csv")

# Separar datos
x = datos.drop('Potabilidad', axis=1)
y = datos['Potabilidad'] 

# Imputación de datos faltantes
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x[:] = imputer.fit_transform(x) 

# Normalización de datos
scaler = StandardScaler()
x = scaler.fit_transform(x)

def build_model():
    model = Sequential()
    model.add(Dense(units=5, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), input_dim=x.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(units=5, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Definir el número de "folds" para la validación cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Inicializar variables para almacenar los resultados
cv_scores = []

# Aplicar la validación cruzada
for train_index, val_index in kf.split(x):
    # Dividir los datos en entrenamiento y validación para cada fold
    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Crear y entrenar el modelo
    model = build_model()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=15, batch_size=100, verbose=0)
    
    # Evaluar el modelo en el fold de validación
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    cv_scores.append(val_acc)
    
    # Graficar el entrenamiento y la validación de precisión
    plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de validación')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Epocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.show()

# Promediar los resultados de la validación cruzada
mean_score = np.mean(cv_scores)
print(f"Precisión promedio de la validación cruzada: {mean_score:.4f}")
