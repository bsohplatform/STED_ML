import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(42)

model = nn.Linear()

'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from bayes_opt import BayesianOptimization
import joblib

bas_csv = pd.read_csv('BAS_DB.csv')
input_list = ['fluid_h','Thi','Tho','mh','dTh','fluid_c','Tci','Tco','mc','dTc','DSH','DSC','comp_eff','Refrigerant',]
bas_data = MinMaxScaler().fit_transform(bas_csv[input_list])
bas_target = bas_csv['COP'].to_numpy()
bas_target = (bas_target - min(bas_target))/(max(bas_target) - min(bas_target))

train_input_full, test_input, train_target_full, test_target = train_test_split(bas_data, bas_target, test_size = 0.05, random_state=42)
train_input, val_input, train_target, val_target = train_test_split(train_input_full, train_target_full, test_size = 0.1, random_state=42)

model = keras.Sequential()
model.add(keras.layers.Dense(10, input_dim=len(input_list), activation='relu'))
model.add(keras.layers.Dense(5, activation='relu'))
model.add(keras.layers.Dense(1))

optimizer = keras.optimizers.RMSprop(lr=1.0e-2)
model.compile(loss='mse',optimizer=optimizer, metrics='accuracy')
history = model.fit(train_input, train_target, epochs=100, verbose=1, validation_data=(val_input, val_target))

print(history)
'''