# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:42:31 2019

@author: ioplab
"""

import numpy as np
import itertools
import cv2


def Datagenerator(x, y, batch_size, input_size, shuffle=False):
    new_idx = np.arange(len(x))
    if shuffle:
        np.random.shuffle(new_idx)
    
    x = np.array(x)[new_idx]
    y = np.array(y)[new_idx]
    
    (H, W) = input_size
    zipped = itertools.cycle(zip(x, y))
    while True:
        X = []
        Y = []
        
        for _ in range(batch_size):
            x, y = zipped.__next__()
            x = cv2.imread(x, 0)
            y = cv2.imread(y, 0)
            x = cv2.resize(x, (H,W)) / 255.0
            y = cv2.resize(y, (H,W)) / 255.0
            X.append(x)
            Y.append(y)
        X = np.array(X).reshape(batch_size,H,W,1)
        Y = np.array(Y).reshape(batch_size,H,W,1)
        yield X, Y