from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py,os
import matplotlib as plt
from sklearn.metrics import roc_curve
from progressbar import ProgressBar
from sklearn.metrics import auc
import matplotlib.pyplot as plt
datadir = os.path.join(os.getcwd(), '..', 'data')

def conv_layer(inputs,filter_num,filter_size,dilation,stride = 1,padding = 'same',initilizer = 'glorot_normal'):
    conv_l = tf.keras.layers.Conv1D(filters = filter_num,
                                    kernel_size = filter_size,
                                    strides = stride,
                                    padding = padding,
                                    dilation_rate = dilation,
                                    kernel_initializer = initilizer
                                    )(inputs)
    conv_l = tf.keras.layers.BatchNormalization()(conv_l)
    conv_l = tf.keras.layers.Activation('relu')(conv_l)
    return conv_l

def residual_block(inputs,filter_num,filter_size,dilation,activation = 'relu'):
        #batch_norm+relu+conv+baatch+relu+conv+residual
        rb_bn1 = tf.keras.layers.BatchNormalization()(inputs)
        rb_ac1 = tf.keras.layers.Activation(activation)(rb_bn1)
        rb_conv1 = conv_layer(rb_ac1,filter_num,filter_size,dilation)
        rb_bn2 = tf.keras.layers.BatchNormalization()(rb_conv1)
        rb_ac2 = tf.keras.layers.Activation(activation)(rb_bn2)
        rb_conv2 = conv_layer(rb_ac2,filter_num,filter_size,dilation)
        residual_sum = tf.keras.layers.add([inputs,rb_conv2])
        return residual_sum
    
def filter_dataset(sequence,target):
    pbar = ProgressBar()
    del_list = []
    for i in pbar(range(0,len(target))):
        if any(y > 0.01 for y in target[i]):
            next
        else:
            del_list.append(i)
    print(len(del_list))
    filtered_seq = np.delete(sequence,del_list,0)
    print(2)
    filtered_target= np.delete(target,del_list,0)
    print(3)
   
    return filtered_seq,filtered_target


#5 output for basic, 21 for major, 
f = h5py.File(datadir+'/output_major.h5','r')
x_test = f['test_seq']
y_test = f['test_label']
x_train = f['train_seq']
y_train = f['train_label']
x_valid = f['valid_seq']
y_valid = f['valid_label']
save_model = datadir+'/model/filtered_model.h5'

f_x_train,f_y_train= filter_dataset(x_train,y_train)
f_x_test,f_y_test= filter_dataset(x_test,y_test)
f_x_valid,f_y_valid= filter_dataset(x_valid,y_valid)


inputs = tf.keras.layers.Input(shape = (4999,4))


conv_1 = conv_layer(inputs,32,5,1)
conv_2 = conv_layer(conv_1,32,5,1)
conv_3 = conv_layer(conv_2,32,5,1)
mp_1 = tf.keras.layers.MaxPool1D()(conv_3)

conv_4 = conv_layer(mp_1,32,5,1)
conv_5 = conv_layer(conv_4,32,5,2)
conv_6 = conv_layer(conv_5,32,5,4)
mp_2 = tf.keras.layers.MaxPool1D()(conv_6)

lstm = tf.keras.layers.LSTM(128)(mp_2)
#flat = tf.keras.layers.Flatten()(mp_2)

dense_1 = tf.keras.layers.Dense(50,activation = 'relu',kernel_initializer='glorot_normal')(lstm)
dense_2 = tf.keras.layers.Dense(50,activation = 'relu',kernel_initializer='glorot_normal')(dense_1)
outputs = tf.keras.layers.Dense(21,activation = 'relu',kernel_initializer='glorot_normal')(dense_2)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.MeanSquaredError()
    
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(inputs = inputs,outputs = outputs,
                optimizer=optimizer,
                loss=loss,
                metrics=['mae'])


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

training = model.fit(f_x_train,f_y_train,
                    epochs=200,
                    batch_size = 64,
                    callbacks = [earlystop,reduce_lr,checkpoint], 
                    validation_data = (f_x_valid,f_y_valid)
                    )
