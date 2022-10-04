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
from tensorflow.keras.layers import BatchNormalization,Dropout
from tensorflow.keras.layers import LeakyReLU,PReLU,ELU,ThresholdedReLU,ReLU
from tensorflow.keras import backend as K


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

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

def deeper_model(nin,optimizer_choice):
    model = keras.Sequential()
    model.add(Dense(100,activation='elu',input_dim=nin))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(120))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(140))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(120))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(100))
    model.add(LeakyReLU())
    model.add(Dense(1,activation='linear'))
    model.compile(
              optimizer=optimizer_choice,
              loss='mean_squared_error',
              metrics=["mean_squared_error", rmse, r_square])
    return model


def simple_model(nin,optimizer_choice):
    model = keras.Sequential()
    model.add(Dense(50,activation='relu',input_dim=nin))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dense(30))
    model.add(ReLU())
    model.add(Dense(1,activation='linear'))
    model.compile(
              optimizer=optimizer_choice,
              loss='mean_squared_error',
              metrics=["mean_squared_error", rmse, r_square])
    return model