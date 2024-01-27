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
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return 1.0 - numerator / (denominator + K.epsilon())

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


class WeightedBinaryCrossEntropy(object):
 
    def __init__(self, pos_ratio):
        neg_ratio = 1. - pos_ratio
        self.pos_ratio = tf.constant(pos_ratio, tf.float32)
        self.weights = tf.constant(neg_ratio / pos_ratio, tf.float32)
        self.__name__ = "weighted_binary_crossentropy({0})".format(pos_ratio)
 
    def __call__(self, y_true, y_pred):
        return self.weighted_binary_crossentropy(y_true, y_pred)
 
    def weighted_binary_crossentropy(self, y_true, y_pred):
        # Transform to logits
        epsilon = tf.convert_to_tensor(K.common._EPSILON, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_pred = tf.log(y_pred / (1 - y_pred))
 
        cost = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.weights)
        return K.mean(cost * self.pos_ratio, axis=-1)


#cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y_, pos_weight=classes_weights))

#POS_WEIGHT = 10  # multiplier for positive targets, needs to be tuned

# def weighted_binary_crossentropy(target, output):#
#     """
#     Weighted binary crossentropy between an output tensor 
#     and a target tensor. POS_WEIGHT is used as a multiplier 
#     for the positive targets.

#     Combination of the following functions:
#     * keras.losses.binary_crossentropy
#     * keras.backend.tensorflow_backend.binary_crossentropy
#     * tf.nn.weighted_cross_entropy_with_logits
#     """
#     # transform back to logits
#     output = tf.clip_by_value(output, K.epsilon, 1 - K.epsilon)
#     output = tf.log(output / (1 - output))
#     # compute weighted loss
#     loss = tf.nn.weighted_cross_entropy_with_logits(targets=target,
#                                                     logits=output,
#                                                     pos_weight=6)
#     return tf.reduce_mean(loss, axis=-1)



def deeper_model(nin,number_of_classes,optimizer_choice,scaling_factor):
    model = keras.Sequential()
    model.add(Input(shape=(nin,)))
    model.add(Dense(50,activation='elu'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(80))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(80))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(40))
    model.add(LeakyReLU())
    model.add(Dense(number_of_classes,activation='softmax'))
    model.compile(
              optimizer=optimizer_choice,
              loss=weighted_categorical_crossentropy(np.array([1,scaling_factor])),
              metrics=['acc',f1_m,precision_m, recall_m,matthews_correlation])
    return model


def simple_model(nin,number_of_classes,optimizer_choice,scaling_factor):
    model = keras.Sequential()
    model.add(Input(shape=(nin,)))
    model.add(Dense(40))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dense(30))
    model.add(ReLU())
    model.add(Dense(number_of_classes,activation='softmax'))
    model.compile(
              optimizer=optimizer_choice,
              loss=weighted_categorical_crossentropy(np.array([1,scaling_factor])),
              metrics=['acc',f1_m,precision_m, recall_m,matthews_correlation])
    return model