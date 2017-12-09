#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 13:09:34 2017

@author: fnord
"""

import math
import numpy
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

from ml_data import SimpleHistData, DinucAutoCovarData, SequenceProteinData, SequenceNucsData, SequenceDinucProperties
from ml_statistics import BaseStatistics


def step_decay(epoch):
    initial_lrate = 0.002
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)


def get_mergerd_model(X):
    from keras.models import Sequential, Model
    from keras.layers import Dense, Merge, Conv2D, MaxPooling2D, Dropout
    from keras.layers import Concatenate, Dense, LSTM, Input, concatenate, Flatten, Bidirectional, TimeDistributed
    from keras.layers.normalization import BatchNormalization
    from keras.layers.embeddings import Embedding
    from keras import backend as K
    
    model = Sequential()
#    model.add(input_shape(X.shape[0], 8, c))
    model.add(Conv2D(filters=50, kernel_size=(3,3) , padding='same', activation='relu', strides=1, input_shape=(X.shape[1], X.shape[2], 1) ) )
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.1))
#    branch2.add(LSTM(32, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
#    model.add(LSTM(32, dropout=0.1, recurrent_dropout=0.1, input_shape=(X.shape[1], )))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation =K.hard_sigmoid))
    
    return model
    
    
    

def get_options():
    from keras import optimizers
    opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    #opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    return opt

# fix random seed for reproducibility
numpy.random.seed(7)


earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', epsilon=0.001, cooldown=0, min_lr=0)

npath = "fasta/Bacillus_non_prom.fa"
ppath = "fasta/Bacillus_prom.fa"
#mldata = SequenceNucsData(npath, ppath, k=3)
mldata = SequenceDinucProperties(npath, ppath)


X = mldata.getX()
Y = mldata.getY()


kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
kf.get_n_splits(X, Y)

cvscores = []
for train_index, test_index in kf.split(X, Y):
    
    X_train = X[train_index,:]
    y_train = Y[train_index]
    
    X_test = X[test_index,:]
    y_test = Y[test_index]
    
    my_const = 100
    X_train *= my_const
    X_test  *= my_const
    
    
    #model = get_model(X_train)
    model = get_mergerd_model(X_train)
    opt = get_options()
    
    loss_function = ['mse', 'binary_crossentropy']
    
    model.compile(loss=loss_function[1], optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    history = model.fit([X_train], y_train, validation_split=0.1, epochs=100, batch_size=64, callbacks=[earlyStopping], class_weight={0:.2, 1:.8})
    print(history.history.keys())
    
    plt.figure()
    plt.ion()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.draw()
    plt.pause(0.001)
    plt.show()
    # summarize history for loss
    plt.figure()
    plt.ion()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.draw()
    plt.pause(0.001)
    plt.show()
    
    Y_pred = model.predict([X_test])
    
    stats = BaseStatistics(y_test, Y_pred)
    print stats
    
    
    scores = model.evaluate([X_test], y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    cvscores.append(scores[1] * 100)
    print scores
    print '='*30

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
print('<<< END >>>')