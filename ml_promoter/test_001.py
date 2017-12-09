#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:21:38 2017

@author: fnord
"""
import math
import numpy
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

from ml_data import SimpleHistData, DinucAutoCovarData, SequenceProteinData, SequenceNucsData
from ml_statistics import BaseStatistics


def step_decay(epoch):
    initial_lrate = 0.002
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)

def get_model(X):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Conv1D, MaxPooling1D
    from keras.layers import LSTM
    from keras.layers import Flatten
    
    max_sample_length = X.shape[1]
    numsamples = X.shape[0]
    
    print 'max_sample_length', max_sample_length
    print 'numsamples', numsamples
    
    model = Sequential()
    model.add(Conv1D(filters=100, kernel_size=5, padding='same', activation='relu', strides=1, input_shape=(max_sample_length,1)))
    #model.add(MaxPooling1D(pool_size=4))
    #model.add(LSTM(80, dropout=0.2, recurrent_dropout=0.1))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def get_mergerd_model(X1, X2, X3):
    from keras.models import Sequential, Model
    from keras.layers import Dense, Merge, Conv1D, MaxPooling1D, Dropout
    from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
    from keras.layers.normalization import BatchNormalization
    from keras.layers.embeddings import Embedding
    
    branch1 = Sequential()
    branch1.add(Dense(X1.shape[1], input_shape=(X1.shape[1], ), init='uniform', activation='relu'))
    branch1.add(Dense(1, init='normal', activation='sigmoid'))
    #branch1.add(BatchNormalization())
    
    branch2 = Sequential()
    branch2.add(Embedding(X2.shape[0], 8, input_length=X2.shape[1]))
    branch2.add(Conv1D(filters=200, kernel_size=9, padding='same', activation='relu', strides=1))
    branch2.add(MaxPooling1D(pool_size=2, strides=2))
#    branch2.add(LSTM(64, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
#    branch2.add(LSTM(32, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
    branch2.add(LSTM(32, dropout=0.1, recurrent_dropout=0.1))
    branch2.add(Dense(X2.shape[1], input_shape=(X2.shape[1], ), init='uniform', activation='relu'))
    branch2.add(Dense(1, init='normal', activation='sigmoid'))
    #branch2.add(BatchNormalization())
    
    branch3 = Sequential()
    branch3.add(Dense(X3.shape[1], input_shape=(X3.shape[1], ), init='uniform', activation='relu'))
    branch3.add(Dense(1, init='normal', activation='sigmoid'))
    #branch3.add(BatchNormalization())
    
    model = Sequential()
    model.add(Merge([branch1, branch2, branch3], mode='concat'))
    #model.add(BatchNormalization())
    model.add(Dense(256, init = 'normal', activation ='sigmoid'))
    model.add(Dropout(0.1))
    model.add(Dense(1, init = 'normal', activation ='sigmoid'))
    
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
mldata = SimpleHistData(npath, ppath, k=4)
mldata2 = SequenceNucsData(npath, ppath, k=3)
mldata3 = DinucAutoCovarData(npath, ppath)

X = mldata.getX()
Y = mldata.getY()

X2 = mldata2.getX()
X3 = mldata3.getX()

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
kf.get_n_splits(X, Y)

cvscores = []
for train_index, test_index in kf.split(X, Y):
    
    X_train = X[train_index,:]
    X_train2 = X2[train_index,:]
    #X_train2 = X_train2[:,:,numpy.newaxis]
    X_train3 = X3[train_index,:]
    
    #X_train = X_train[:,:,numpy.newaxis]
    #X_train = numpy.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    y_train = Y[train_index]
    
    X_test = X[test_index,:]
    X_test2 = X2[test_index,:]
    #X_test2 = X_test2[:,:,numpy.newaxis]
    
    X_test3 = X3[test_index,:]
    #X_test = X_test[:,:,numpy.newaxis]
    #X_test = numpy.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_test = Y[test_index]
    
    
#    lenX1 = X_train.shape[1]
#    lenX2 = X_train2.shape[1]
#    lenX3 = X_train3.shape[1]
#    sm = SMOTE(random_state=42, ratio='minority', kind='borderline1')
#    X_tmp = numpy.hstack((X_train, X_train2, X_train3))
#    X_res, y_res = sm.fit_sample(X_tmp, y_train)
#    X_train = X_res[:, 0:lenX1]
#    assert X_train.shape[1]==lenX1
#    X_train2 = X_res[:, lenX1:(lenX1+lenX2)]
#    assert X_train2.shape[1]==lenX2
#    X_train3 = X_res[:, (lenX1+lenX2):]
#    assert X_train3.shape[1]==lenX3
    
    #model = get_model(X_train)
    model = get_mergerd_model(X_train, X_train2, X_train3)
    opt = get_options()
    
    loss_function = ['mse', 'binary_crossentropy']
    
    model.compile(loss=loss_function[1], optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    history = model.fit([X_train, X_train2, X_train3], y_train, validation_split=0.1, epochs=100, batch_size=64, callbacks=[earlyStopping], class_weight={0:.2, 1:.8})
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
    
    Y_pred = model.predict([X_test, X_test2, X_test3])
    
    stats = BaseStatistics(y_test, Y_pred)
    print stats
    
    
    scores = model.evaluate([X_test, X_test2, X_test3], y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    cvscores.append(scores[1] * 100)
    print scores
    print '='*30

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
print('<<< END >>>')