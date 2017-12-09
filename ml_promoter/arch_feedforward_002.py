#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 12:26:35 2017

@author: fnord
"""

from utils import join_data
from utils import get_input_lengths
from utils import calc_limits
from utils import crop_data
from utils import get_model_data_full

# GLOBALS
shared_input_length = None
data = None
limits = None
lengths = None

    
def setup_data(npath, ppath):
    from ml_data import SequenceDinucProperties
    from ml_data import SimpleHistData
    
    global data
    global limits
    global lengths
    global shared_input_length
    
    # Join data into a single input vector
    data = join_data(
            SimpleHistData(npath, ppath, k=3)      
    )
    
    # Get lenghts of input vectors
    lengths = get_input_lengths(data)
    
    # Define limits of data type on single vector
    limits = calc_limits(data)
    
    # Compute shared input vector length
    shared_input_length = 64
    

def create_model():
    from keras import backend as K
    from keras.models import Model
    from keras.models import Sequential
    from keras.layers import Input
    from keras.layers import Dense
    from keras.layers import Conv2D
    from keras.layers import MaxPool2D
    from keras.layers import Flatten
    from keras.layers import Dropout
    from keras.layers import LSTM
    from keras.layers import TimeDistributed
    from keras.layers import concatenate
    
    # Set Shared Input
    shared_input = Input(shape=(shared_input_length, ), dtype='float32', name='input')
    

#    flat = Flatten(name='flatten')(shared_input)
    
#    hidden = Dense(128, activation=K.sigmoid, name='hidden')(shared_input)
    
    # Add last neuron
#            prob = Dense(1, activation=K.sigmoid, kernel_initializer=K.random_normal)
    prob = Dense(1, activation=K.sigmoid, name='prob')(shared_input)
    
    
    # Setup model object
    model = Model(inputs=shared_input, outputs=prob)            
    return model

def run(npath, ppath):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.utils import plot_model
    from keras.callbacks import EarlyStopping
    from ml_statistics import BaseStatistics
    from keras import optimizers
    import numpy as np
    
    
    
    # Params
    weightTrue = 0.8
    class_weight = {0:(1-weightTrue), 1:weightTrue}
    earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
#    opt = optimizers.RMSprop(lr=0.001, decay=0.0)
    
    # Setup all data for inputs
    setup_data(npath, ppath)
    
    # Retrieve data for model
    X, Y = get_model_data_full(data)
    
    X = X*100000
    
    print X
    print Y.shape
    
#    plot_model(create_model(), show_shapes=True)
    
    # evaluate using 10-fold cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
#    results = cross_val_score(model, X, Y, cv=kfold)
#    print('='*30)
#    print(results.mean())
    
    kfold.get_n_splits(X, Y)
    
    for train_index, test_index in kfold.split(X, Y):
        # TRAIN DATA
        X_train, y_train = X[train_index,:], Y[train_index]
        # TEST DATA
        X_test, y_test = X[test_index,:], Y[test_index]
        
        # create model
#        model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=32, verbose=1)
        model = create_model()
        
        
        # Compile Model
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        print(model.summary())
        
        history = model.fit(X_train, y_train, validation_split=0.3, epochs=100, batch_size=128, callbacks=[earlyStopping], class_weight=class_weight, verbose=0)
        print(history.history.keys())
        
        Y_pred = model.predict(X_test, verbose=0)
    
        stats = BaseStatistics(y_test, Y_pred)
        print stats
    

npath = "fasta/Bacillus_non_prom.fa"
ppath = "fasta/Bacillus_prom.fa"

#run_grid(npath, ppath)
run(npath, ppath)
