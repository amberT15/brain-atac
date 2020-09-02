from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py,os
import model_structure
import matplotlib as plt
import sklearn.metrics
from sklearn.metrics import roc_curve
from progressbar import ProgressBar
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt



def conv_layer(inputs,filter_num,filter_size,dilation,stride = 1,padding = 'same',initilizer = 'glorot_normal',activation = 'relu'):
    #one convolutional layer with batch norm and activation
    conv_l = tf.keras.layers.Conv1D(filters = filter_num,
                                    kernel_size = filter_size,
                                    strides = stride,
                                    padding = padding,
                                    dilation_rate = dilation,
                                    kernel_initializer = initilizer
                                    )(inputs)
    conv_l = tf.keras.layers.BatchNormalization()(conv_l)
    conv_l = tf.keras.layers.Activation(activation)(conv_l)
    return conv_l

def residual_block(inputs,filter_num,filter_size,dilation,activation = 'relu'):
    #batch_norm+relu+conv+batch+relu+conv+residual
    rb_bn1 = tf.keras.layers.BatchNormalization()(inputs)
    rb_ac1 = tf.keras.layers.Activation(activation)(rb_bn1)
    rb_conv1 = conv_layer(rb_ac1,filter_num,filter_size,dilation)
    rb_bn2 = tf.keras.layers.BatchNormalization()(rb_conv1)
    rb_ac2 = tf.keras.layers.Activation(activation)(rb_bn2)
    rb_conv2 = conv_layer(rb_ac2,filter_num,filter_size,dilation)
    residual_sum = tf.keras.layers.add([inputs,rb_conv2])
    return residual_sum
    
def filter_dataset(target,limit):
    #only include reads with at least one peak
    #limit = filter threshold
    binary_target = np.zeros(len(target))
    pbar = ProgressBar()
    del_list = []
    for i in pbar(range(0,len(target))):
        if target[i] > limit:
            next
        else:
            binary_target[i] = 1
    
    return binary_target

def early_stopping(patience = 20, verbose = 1):
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                             min_delta=0, 
                                             patience=patience, 
                                             verbose=verbose, 
                                             mode='min', 
                                             baseline=None, restore_best_weights=True)
    return earlystop
    
    
    
def model_checkpoint(save_path,save_best_only = True):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path, 
                                                monitor='val_loss', 
                                                verbose=1, 
                                                save_best_only=save_best_only, 
                                                mode='min')
    return checkpoint
    
def reduce_lr(patience = 3):
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 factor=0.5,
                                                 patience=3, 
                                                 min_lr=1e-7,
                                                 mode='min',
                                                 verbose=1)
    return reduce_lr

def curve_figure(y_pred,y_test,exp_count):
    
    #for each experiment
    pred = y_pred
    test = y_test
    total = len(pred)
    
    #plot roc curve
    fpr,tpr,roc_thr = sklearn.metrics.roc_curve(test,pred)
    auroc = sklearn.metrics.auc(fpr,tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auroc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show();
    plt.close()
    
    #plot pr curve
    precision, recall, pr_thr = sklearn.metrics.precision_recall_curve(test, pred)
    aupr = sklearn.metrics.auc(recall,precision)
    plt.plot(recall, precision, label='Keras (area = {:.3f})'.format(aupr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.show();
    plt.close()
        

def CNN_dense(input_size,label_num,padding = 'same',activation = 'relu'):
    
    inputs = tf.keras.layers.Input(shape = input_size)
    
    conv_1 = conv_layer(inputs,32,5,1,activation = activation)
    conv_2 = conv_layer(conv_1,32,5,1,activation = activation)
    conv_3 = conv_layer(conv_2,32,5,1,activation = activation)
    mp_1 = tf.keras.layers.MaxPool1D()(conv_3)

    conv_4 = conv_layer(mp_1,32,5,1,activation = activation)
    conv_5 = conv_layer(conv_4,32,5,1,activation = activation)
    conv_6 = conv_layer(conv_5,32,5,1,activation = activation)
    mp_2 = tf.keras.layers.MaxPool1D()(conv_6)

    flat = tf.keras.layers.Flatten()(mp_2)

    dense_1 = tf.keras.layers.Dense(50,activation = activation,kernel_initializer='glorot_normal')(flat)
    dense_2 = tf.keras.layers.Dense(50,activation = activation,kernel_initializer='glorot_normal')(dense_1)
    outputs = tf.keras.layers.Dense(1,activation ='sigmoid',kernel_initializer='glorot_normal')(dense_2)

    auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(inputs = inputs,outputs = outputs,
                    optimizer=optimizer,
                    loss=loss,
                     metrics = ['accuracy',auroc,aupr])

    return model

def CNN_LSTM(input_size,label_num,padding = 'same',activation = 'relu'):
    
    inputs = tf.keras.layers.Input(shape = input_size)
    
    conv_1 = conv_layer(inputs,32,5,1,activation = activation)
    conv_2 = conv_layer(conv_1,32,5,1,activation = activation)
    conv_3 = conv_layer(conv_2,32,5,1,activation = activation)
    mp_1 = tf.keras.layers.MaxPool1D()(conv_3)

    conv_4 = conv_layer(mp_1,32,5,1,activation = activation)
    conv_5 = conv_layer(conv_4,32,5,1,activation = activation)
    conv_6 = conv_layer(conv_5,32,5,1,activation = activation)
    mp_2 = tf.keras.layers.MaxPool1D()(conv_6)

    lstm = tf.keras.layers.LSTM(128)(mp_2)

    dense_1 = tf.keras.layers.Dense(50,activation = activation,kernel_initializer='glorot_normal')(lstm)
    dense_2 = tf.keras.layers.Dense(50,activation = activation,kernel_initializer='glorot_normal')(dense_1)
    outputs = tf.keras.layers.Dense(1,activation ='sigmoid',kernel_initializer='glorot_normal')(dense_2)


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(inputs = inputs,outputs = outputs,
                    optimizer=optimizer,
                    loss=loss,
                     metrics = ['accuracy',auroc,aupr])
    
    return model
    
def Residual_dense(input_shape,label_num,padding = 'same',activation = 'relu'):
    
    inputs = tf.keras.layers.Input(shape = input_shape)
    
    conv_1 = tf.keras.layers.Conv1D(32,11,1,kernel_initializer = 'glorot_normal',
                                      padding=padding)(inputs)
    rb_1 = residual_block(conv_1,32,11,1)
    rb_2 = residual_block(rb_1,32,11,2)
    rb_3 = residual_block(rb_1,32,11,4)
    bn = tf.keras.layers.BatchNormalization()(rb_3)
    ac = tf.keras.layers.Activation(activation)(bn)
    
    conv_2 = conv_layer(ac,5,1,1)
    
    flat = tf.keras.layers.Flatten()(conv_2)

    dense_1 = tf.keras.layers.Dense(50,activation = activation,kernel_initializer='glorot_normal')(flat)
    dense_2 = tf.keras.layers.Dense(50,activation = activation,kernel_initializer='glorot_normal')(dense_1)
    outputs = tf.keras.layers.Dense(1,activation ='sigmoid',kernel_initializer='glorot_normal')(dense_2)


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss = tf.keras.losses.BinaryCrossentropy()
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(inputs = inputs,outputs = outputs,
                    optimizer=optimizer,
                    loss=loss,
                     metrics = ['accuracy',auroc,aupr])
    
    return model
    
#def Residual_LSTM():
