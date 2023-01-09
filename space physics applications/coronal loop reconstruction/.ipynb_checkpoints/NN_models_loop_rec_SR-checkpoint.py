# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 08:41:56 2021

@author: savvra
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
from tensorflow.keras.layers import BatchNormalization,Dropout,Conv2D, Conv1D, Flatten
from tensorflow.keras.layers import LeakyReLU,PReLU,ELU,ThresholdedReLU,ReLU, MaxPooling1D
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


def XGboost_Convmodel(batch_size,optimizer_choice):
    model = keras.Sequential()
    model.add(Conv1D(batch_size,(100,), activation='relu', input_shape=(1500,5)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(batch_size,(50,), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(batch_size,(25,), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(batch_size,(10,), activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(6000, activation='relu',name='XGboost_exporter'))
    model.add(Dense(1500, activation='linear'))
    model.compile(
              optimizer=optimizer_choice,
              loss='mean_squared_error',
              metrics=["mean_squared_error", rmse, r_square])
    return model

def deep_Conv1D_model(batch_size,optimizer_choice):
    model = keras.Sequential()
    model.add(Conv1D(batch_size,(100,), activation='relu', input_shape=(1500,5)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(batch_size,(50,), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(batch_size,(25,), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(batch_size,(10,), activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(6000, activation='relu'))
    model.add(Dense(1500, activation='linear'))
    model.compile(
              optimizer=optimizer_choice,
              loss='mean_squared_error',
              metrics=["mean_squared_error", rmse, r_square])
    return model