from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py,os

import model_structure
import importlib
importlib.reload(model_structure)

import matplotlib as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from progressbar import ProgressBar
from sklearn.metrics import auc
import matplotlib.pyplot as plt
datadir = '/mnt/1a18a49e-9a31-4dbf-accd-3fb8abbfab2d/brain_atac/data'

#5 output for basic, 21 for major, 
f = h5py.File(datadir+'/output_basic.h5','r')
x_test = f['test_seq']
y_test = f['test_label']
x_train = f['train_seq']
y_train = f['train_label']
x_valid = f['valid_seq']
y_valid = f['valid_label']
save_model = datadir+'/model/filtered_model.h5'

b_y_train= model_structure.filter_dataset(y_train[:,0],0.05)
b_y_test= model_structure.filter_dataset(y_test[:,0],0.05)
b_y_valid= model_structure.filter_dataset(y_valid[:,0],0.05)

model = model_structure.CNN_dense((4999,4),1)

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                             min_delta=0, 
                                             patience=20, 
                                             verbose=1, 
                                             mode='min', 
                                             baseline=None, restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 factor=0.5,
                                                 patience=5, 
                                                 min_lr=1e-7,
                                                 mode='min',
                                                 verbose=1)

checkpoint = tf.keras.callbacks.ModelCheckpoint(save_model, 
                                                monitor='val_loss', 
                                                verbose=1, 
                                                save_best_only=True, 
                                                mode='min')
training = model.fit(x_train[()],b_y_train,
                    epochs=200,
                    batch_size = 64,
                    callbacks = [earlystop,reduce_lr,checkpoint], 
                    validation_data = (x_valid[()],b_y_valid)
                    )



