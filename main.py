# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 23:18:10 2019

@author: jacky
"""

from data import Datagenerator
from model import SRRNN
from loss import psnr
from keras import callbacks
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
np.random.seed(7)
#%%
def showHistory(history, savepath, legend_loc='upper right', *keys): #  legend_loc = 'lower right' & 'upper right'
    for key in keys:
        plt.plot(history[key])
    xtks = np.linspace(0, len(history[key])-1, 8, dtype='int32')
    xlbl = np.linspace(1, len(history[key]), 8, dtype='int32')
    plt.xticks(xtks,xlbl)
    plt.title('Train History')
    plt.xlabel('Epoch')
    plt.legend(keys, loc=legend_loc)
    if savepath != None:
        plt.savefig(savepath)
    plt.show()
#%%
tra_X = sorted(glob.glob(r'D:\YJ\database\dogs-vs-cats\tra_blur\*.jpg'))[:]
tra_Y = sorted(glob.glob(r'D:\YJ\database\dogs-vs-cats\tra_target\*.jpg'))[:]

tes_X = sorted(glob.glob(r'D:\YJ\database\dogs-vs-cats\tes_blur\*.jpg'))[:]
tes_Y = sorted(glob.glob(r'D:\YJ\database\dogs-vs-cats\tes_target\*.jpg'))[:]

G_tra = Datagenerator(tra_X, tra_Y, 4, (256,256))
G_val = Datagenerator(tes_X, tes_Y, 4, (256,256))

model = SRRNN()
model.compile(optimizer=Adam(lr=0.0001), loss='MSE', metrics=['MSE','MAE',psnr])
hist = model.fit_generator(generator=G_tra,
                           steps_per_epoch=25000/4, 
                           validation_data=G_val,
                           validation_steps=12500/4,
                           initial_epoch=0, epochs=45,
                           callbacks=[callbacks.ModelCheckpoint('model.h5',
                                                                monitor='val_loss', verbose=1, mode='min',
                                                                save_best_only=True, save_weights_only=True),
                                      callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)])
showHistory(hist.history, None, 'upper right', 'loss', 'val_loss')
showHistory(hist.history, None, 'upper right', 'loss', 'val_loss')
#%%
G_tes = Datagenerator(tes_X, tes_Y, 4, (256,256))
model.load_weights('model.h5')
pred = model.predict_generator(generator=G_tes, steps=12500/4, verbose=1)
#%%
idx = 1
x = cv2.resize(cv2.imread(tes_X[idx], 0), (256,256)) / 255.0
y = cv2.resize(cv2.imread(tes_Y[idx], 0), (256,256)) / 255.0
pr = np.clip(pred[idx,:,:,0], 0.0, 1.0)

plt.imshow(x, cmap='gray'); plt.colorbar(); plt.show()
plt.imshow(y, cmap='gray'); plt.colorbar(); plt.show()
plt.imshow(pr, cmap='gray'); plt.colorbar(); plt.show()

cv2.imwrite(r'D:\YJ\database\dogs-vs-cats\0_x.jpg', x*255.0)
cv2.imwrite(r'D:\YJ\database\dogs-vs-cats\1_y.jpg', y*255.0)
cv2.imwrite(r'D:\YJ\database\dogs-vs-cats\2_pr.jpg', pr*255.0)