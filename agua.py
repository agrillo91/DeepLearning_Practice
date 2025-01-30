# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datos = pd.read_csv("potabilidad_agua.csv")

x = datos.drop('Potabilidad', axis=1)
y = datos['Potabilidad']
