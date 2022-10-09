# -*- coding: utf-8 -*-
"""
@author: Savvas Raptis

"""
import os
import tensorflow as tf
import datetime
from os import path
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization,Dropout
from tensorflow.keras.layers import LeakyReLU,PReLU,ELU,ThresholdedReLU,ReLU
from tensorflow.keras import backend as K

# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
    from tensorflow.keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression
def mse(y_true, y_pred):
    from tensorflow.keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from tensorflow.keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def r_square_loss(y_true, y_pred):
    from tensorflow.keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))

def simple_model(nin,optimizer_choice):
    model = keras.Sequential()
    model.add(Dense(50,activation='relu',input_dim=nin))
    model.add(ReLU())
    model.add(Dense(30))
    model.add(ReLU())
    model.add(Dense(1,activation='linear'))
    model.compile(
              optimizer=optimizer_choice,
              loss='mean_squared_error',
              metrics=["mean_squared_error", rmse, r_square])
    return model