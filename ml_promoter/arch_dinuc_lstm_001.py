#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 13:09:34 2017

@author: fnord
"""

import math
import numpy
import matplotlib.pyplot as plt
from time import time

from sklearn.model_selection import StratifiedKFold

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.callbacks import TensorBoard

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

initializers = ['glorot_uniform', 'lecun_uniform', 'uniform']


def get_mergerd_model(X1, X2):
    from keras.models import Sequential
    from keras.models import Model
    from keras.layers import Dense
    from keras.layers import Merge
    from keras.layers import Conv1D
    from keras.layers import MaxPooling1D
    from keras.layers import AveragePooling1D
    from keras.layers import Dropout
    from keras.layers import Flatten
    from keras.layers import TimeDistributed
    from keras.layers import Bidirectional
    from keras.layers import LSTM
    from keras.layers import Concatenate
    from keras.layers.normalization import BatchNormalization
    from keras.layers.embeddings import Embedding
    from keras import backend as K
    from keras.constraints import maxnorm

    vector_length_di = 32
    modelDi = Sequential()    
    modelDi.add(Embedding(16, vector_length_di, input_length=X1.shape[1]))    
    modelDi.add(Conv1D(filters=300, kernel_size=9, strides=1, activation='relu', kernel_initializer=initializers[0]))
    modelDi.add(MaxPooling1D(pool_size=3, strides=1))
#    modelDi.add(Flatten())
    modelDi.add(Bidirectional(LSTM(50, dropout=0.2, recurrent_dropout=0.3)))
#    modelDi.add(Bidirectional(LSTM(50, dropout=0.1, recurrent_dropout=0.2, return_sequences=True)))
#    modelDi.add(TimeDistributed(Dense(1, activation=K.sigmoid, name='DDi')))
    
    vector_length_tri = 64   
    modelTri = Sequential()    
    modelTri.add(Embedding(64, vector_length_tri, input_length=X2.shape[1]))
    modelTri.add(Conv1D(filters=300, kernel_size=9, strides=1, activation='relu', kernel_initializer=initializers[0]))
    modelTri.add(MaxPooling1D(pool_size=3, strides=1))
#    modelTri.add(Flatten())
    modelTri.add(Bidirectional(LSTM(50, dropout=0.2, recurrent_dropout=0.3)))
#    modelTri.add(Bidirectional(LSTM(50, dropout=0.1, recurrent_dropout=0.2, return_sequences=True)))
#    modelTri.add(TimeDistributed(Dense(1, activation=K.sigmoid, name='DTri')))
    
    
    last = Sequential()    
    last.add(Merge([modelDi, modelTri], mode='concat', concat_axis=-1))  
#    last.add(Concatenate([modelDi, modelTri]))
    last.add(Dropout(.2))    
    last.add(Dense(1, activation=K.sigmoid, name='DLast'))
    
#    model.add(TimeDistributed(Flatten()))
#    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)))
#    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.1)))
#    branch2.add(LSTM(32, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
#    model.add(LSTM(32, dropout=0.1, recurrent_dropout=0.1, input_shape=(X.shape[1], )))
#    model.add(Dense(128, activation=K.sigmoid))
    
    return last
    
    
    

def get_options():
    from keras import optimizers
    opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
#    opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    opt = 'adam'
    return opt

# fix random seed for reproducibility
numpy.random.seed(7)


earlyStopping=EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=5, verbose=0, mode='auto', epsilon=0.001, cooldown=0, min_lr=0)

#npath = "fasta/Bacillus_non_prom.fa"
#ppath = "fasta/Bacillus_prom.fa"

#npath = "fasta/Arabidopsis_non_prom_big.fa"
#ppath = "fasta/Arabidopsis_non_tata.fa"

npath = "fasta/Ecoli_non_prom.fa"
ppath = "fasta/Ecoli_prom.fa"

mldata = SequenceNucsData(npath, ppath, k=2)
mldata2 = SequenceNucsData(npath, ppath, k=3)

X = mldata.getX()
X2 = mldata2.getX()
newCol = numpy.array([-1 for _ in range(X2.shape[0])])
X2 = numpy.column_stack([ X2, newCol ])
Y = mldata.getY()

print 'X1: {}'.format(X.shape)
print 'X2: {}'.format(X2.shape)
print 'Y: {}'.format(Y.shape)


toRemove = False
if toRemove:
    posIndex = numpy.where( Y[:]==1 )[0]
    negIndex = numpy.where( Y[:]==0 )[0]
    
    diff = len(negIndex)-len(posIndex)
    diff = len(negIndex)-diff
    print 'Pos: {} | Neg; {} | DIFF: {}'.format(len(posIndex), len(negIndex), diff)
    
    toremove = numpy.arange(len(negIndex))
    numpy.random.shuffle(toremove)
    toremove=toremove[:diff]
    
    X = numpy.delete(X, toremove, 0)
    X2 = numpy.delete(X2, toremove, 0)
    Y = numpy.delete(Y, toremove, 0)
    print 'Pos remove shape = X {} | Y {}'.format(X.shape, Y.shape)


kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
kf.get_n_splits(X, Y)

cvscores = []
cnt = 0

numpy.random.seed(123556)    

for train_index, test_index in kf.split(X, Y):
    
    X_train = X[train_index,:]
    X2_train = X2[train_index,:]
    y_train = Y[train_index]
    
    X_test = X[test_index,:]
    X2_test = X2[test_index,:]
    y_test = Y[test_index]    
    
    if cnt==0:
        print( len(y_train[ numpy.where(y_train[:]==0) ]), len(y_train[ numpy.where(y_train[:]==1) ]) )
        print( len(y_test[ numpy.where(y_test[:]==0) ]), len(y_test[ numpy.where(y_test[:]==1) ]) )

    model = get_mergerd_model(X_train, X2_train)
    opt = get_options()
    
    loss_function = ['mse', 'binary_crossentropy']
    
    model.compile(loss=loss_function[1], optimizer=opt, metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    if cnt==0:
        print(model.summary())
        
#    calls = [tensorboard, earlyStopping, reduceLR]
    calls = [tensorboard, earlyStopping]
    weights={0:.3, 1:.7}
    history = model.fit([X_train, X2_train], y_train, validation_split=0.1, epochs=250, batch_size=32, callbacks=calls, class_weight=weights, verbose=0)
    
    Y_pred = model.predict([X_test, X2_test])
    
    stats = BaseStatistics(y_test, Y_pred)
    print stats
    
    
    scores = model.evaluate([X_test, X2_test], y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    cvscores.append(scores[1] * 100)
    print scores
    print '='*30
    cnt+=1

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
print('<<< END >>>')