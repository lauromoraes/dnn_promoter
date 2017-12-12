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
    from keras.models import Sequential, Model
    from keras.layers import Dense, Merge, Conv2D, MaxPooling2D, Dropout, AveragePooling2D
    from keras.layers import Dense, LSTM, Flatten, TimeDistributed, Bidirectional
    from keras.layers.normalization import BatchNormalization
    from keras.layers.embeddings import Embedding
    from keras import backend as K
    from keras.constraints import maxnorm

    model = Sequential()
#    model.add(input_shape(X.shape[0], 8, c))
    model.add(Conv2D(filters=300, kernel_size=(X1.shape[1],9), kernel_initializer=initializers[0],  activation='relu', strides=(1, 3), input_shape=(X1.shape[1], X1.shape[2], X1.shape[3] ) ) )
    model.add(MaxPooling2D(pool_size=(1,3), strides=(1,1)))
    model.add(Flatten())
#    model.add(TimeDistributed(Flatten()))
#    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)))
#    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.1)))
#    branch2.add(LSTM(32, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
#    model.add(LSTM(32, dropout=0.1, recurrent_dropout=0.1, input_shape=(X.shape[1], )))
#    model.add(Dense(128, activation=K.sigmoid))
    
    hist = Sequential()
    hist.add(Dense(X2.shape[1], input_shape=(X2.shape[1], )))
    
    
    last = Sequential()
    
    last.add(Merge([model, hist], mode='concat'))
    
    last.add(Dropout(.1))
    
    last.add(Dense(1, activation=K.sigmoid))
    
    return last
    
    
    

def get_options():
    from keras import optimizers
    opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
#    opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    return opt

# fix random seed for reproducibility
numpy.random.seed(7)


earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', epsilon=0.001, cooldown=0, min_lr=0)

npath = "fasta/Bacillus_non_prom.fa"
ppath = "fasta/Bacillus_prom.fa"

#npath = "fasta/Ecoli_non_prom.fa"
#ppath = "fasta/Ecoli_prom.fa"

#mldata = SequenceNucsData(npath, ppath, k=3)
mldata = SequenceDinucProperties(npath, ppath)
mldata2 = SimpleHistData(npath, ppath, k=4, upto=True)


X = mldata.getX()
X2 = mldata2.getX()
Y = mldata.getY()

print X.shape
print Y.shape

posIndex = numpy.where( Y[:]==1 )[0]
negIndex = numpy.where( Y[:]==0 )[0]

diff = len(negIndex)-len(posIndex)
diff = len(negIndex)-diff
print 'DIFF', diff

toremove = numpy.arange(len(negIndex))
numpy.random.shuffle(toremove)
toremove=toremove[:diff]

X = numpy.delete(X, toremove, 0)
X2 = numpy.delete(X2, toremove, 0)
Y = numpy.delete(Y, toremove, 0)
print 'shape'
print X.shape
print Y.shape


kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
kf.get_n_splits(X, Y)

cvscores = []
cnt = 0

seed = numpy.random.randint(0,100000,1)[0]
print 'SEED', seed
numpy.random.seed(123556)    
props = numpy.random.randint(0,2,38)
print 'NUM SELECTEDS', numpy.sum(props)
print props
props = [ True if x==1 else False for x in props ]

for train_index, test_index in kf.split(X, Y):
    
    X_train = X[train_index,:]
    X2_train = X2[train_index,:]
    y_train = Y[train_index]
    
    X_test = X[test_index,:]
    X2_test = X2[test_index,:]
    y_test = Y[test_index]
    
    
    
    X_train = X_train[:,props,:,:]
    X_test = X_test[:,props,:,:]
    
    print X_train.shape
    
    
    if cnt==0:
        print( len(y_train[ numpy.where(y_train[:]==0) ]), len(y_train[ numpy.where(y_train[:]==1) ]) )
        print( len(y_test[ numpy.where(y_test[:]==0) ]), len(y_test[ numpy.where(y_test[:]==1) ]) )
    
    
    #model = get_model(X_train)
    model = get_mergerd_model(X_train, X2_train)
    opt = get_options()
    
    loss_function = ['mse', 'binary_crossentropy']
    
    model.compile(loss=loss_function[1], optimizer=opt, metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    if cnt==0:
        print(model.summary())
    history = model.fit([X_train, X2_train], y_train, validation_split=0.2, epochs=250, batch_size=64, callbacks=[tensorboard,earlyStopping,reduceLR], class_weight={0:.3, 1:.7}, verbose=0)
#    print(history.history.keys())
    
#    plt.figure()
#    plt.ion()
#    plt.plot(history.history['acc'])
#    plt.plot(history.history['val_acc'])
#    plt.title('model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
#    plt.draw()
#    plt.pause(0.001)
#    plt.show()
#    # summarize history for loss
#    plt.figure()
#    plt.ion()
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
#    plt.draw()
#    plt.pause(0.001)
#    plt.show()
    
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