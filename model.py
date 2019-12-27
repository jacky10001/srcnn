# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:39:23 2019

@author: jacky
"""

from keras import layers
from keras import models


def down_block(x,filters,kernel_size=(3,3),strides=(1,1)):
    x1 = layers.Conv2D(filters, 
               kernel_size=kernel_size, 
               strides=strides, 
               padding='same', # same or valid
               # activation=None, 
               # use_bias=True, 
               kernel_initializer='glorot_uniform', 
               bias_initializer='zeros', 
               kernel_regularizer=None, 
               bias_regularizer=None, 
               activity_regularizer=None)(x)
    x = layers.BatchNormalization(axis=-1)(x1)
    x3 = layers.Activation('relu')(x)
    return layers.add([x1,x3])

def up_block(x,filters,kernel_size=(3,3),strides=(2,2)):
    x1 = layers.Conv2DTranspose(filters,
                                kernel_size=kernel_size,
                                strides=strides, 
                                padding='same',
                                # use_bias=True, 
                                kernel_initializer='glorot_uniform', 
                                bias_initializer='zeros', 
                                kernel_regularizer=None, 
                                bias_regularizer=None, 
                                activity_regularizer=None)(x)
    x = layers.BatchNormalization(axis=-1)(x1)
    x3 = layers.Activation('relu')(x)
    return layers.add([x1,x3])

def SRRNN():
    x = layers.Input(shape=(256,256,1))
    d1 = down_block(x,16)
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(d1)
    
    d2 = down_block(p1,32)
    d2 = down_block(d2,32)
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(d2)
    
    d3 = down_block(p2,48)
    d3 = down_block(d3,48)
    p3 = layers.MaxPooling2D(pool_size=(2, 2))(d3)
    
    d4 = down_block(p3,64)
    d4 = down_block(d4,64)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(d4)
    
    d5 = down_block(p4,96)
    d5 = down_block(d5,96)
    p5 = layers.MaxPooling2D(pool_size=(2, 2))(d5)
    
    d6 = down_block(p5,128)
    p6 = layers.MaxPooling2D(pool_size=(2, 2))(d6)
    
    d7 = layers.add([up_block(p6,128),d6])
    d8 = layers.add([up_block(d7,96),d5])
    d9 = layers.add([up_block(d8,64),d4])
    d10 = layers.add([up_block(d9,48),d3])
    d11 = layers.add([up_block(d10,32),d2])
    d12 = layers.add([up_block(d11,16),d1])
    
    y = layers.Conv2D(1, kernel_size=(3,3), strides=(1,1), padding='same',
                      kernel_initializer='glorot_uniform', bias_initializer='zeros')(d12)
    
    model = models.Model(inputs=x, outputs=y)
    model.summary()
    return model