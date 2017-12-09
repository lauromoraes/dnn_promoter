#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:40:02 2017

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
    
    global data
    global limits
    global lengths
    global shared_input_length
    
    # Join data into a single input vector
    data = join_data(
            SequenceDinucProperties(npath, ppath)
    )
    
    # Get lenghts of input vectors
    lengths = get_input_lengths(data)
    
    # Define limits of data type on single vector
    limits = calc_limits(data)
    
    # Compute shared input vector length
    shared_input_length = 79
    

def create_model(optimizer='nadam', activation='sigmoid'):
    from keras import backend as K
    from keras.models import Model
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
    shared_input = Input(shape=(38, shared_input_length,1), dtype='float32', name='input')
    
    conv1 = Conv2D(filters=200, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', name='conv2d')(shared_input)
    
    pool = MaxPool2D(pool_size=(1,3), name='max2d')(conv1)
    

    flat = Flatten(name='flatten')(pool)
#    flat = TimeDistributed(Flatten(name='flatten'))(pool)
#    
#    lstm = LSTM(32, name='lstm')(flat)
    
    drop = Dropout(.1)(flat)
    
#    hidden = Dense(128, activation=K.sigmoid, name='hidden')(drop)
    
    # Add last neuron
#            prob = Dense(1, activation=K.sigmoid, kernel_initializer=K.random_normal)
    prob = Dense(1, activation=K.sigmoid, name='prob')(drop)
    
    
    # Setup model object
    model = Model(inputs=shared_input, outputs=prob)            

    return model

def define_grid():
    activation = ['hard_sigmoid']
    
    grid_params = dict(
                activation=activation
            )
    return grid_params

def run_grid(npath, ppath, weightTrue=0.75):
    from sklearn.model_selection import GridSearchCV
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.utils import plot_model
    
    weightTrue = 0.8
    class_weight = {0:(1-weightTrue), 1:weightTrue}
    
    # Setup all data for inputs
    setup_data(npath, ppath)
    
    # Retrieve data for model
    X, Y = get_model_data_full(data)
    
    # Define parameters to grid search
    param_grid = define_grid()
    
    # create model
    model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=64, verbose=1)
    
    
    plot_model(create_model(), show_shapes=True)
    
    # Setup grid object
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1 )
    
    # Fit model and find scores
    grid_result = grid.fit(X, Y, validation_split=0.1, class_weight=class_weight)
    
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def run(npath, ppath):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.utils import plot_model
    from keras.callbacks import EarlyStopping
    from ml_statistics import BaseStatistics
    from keras import optimizers
    
    
    
    # Params
    weightTrue = 0.8
    class_weight = {0:(1-weightTrue), 1:weightTrue}
    earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    opt = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
#    opt = optimizers.RMSprop(lr=0.001, decay=0.0)
    
    # Setup all data for inputs
    setup_data(npath, ppath)
    
    # Retrieve data for model
    X, Y = get_model_data_full(data)
    
#    plot_model(create_model(), show_shapes=True)
    
    # evaluate using 10-fold cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1543)
#    results = cross_val_score(model, X, Y, cv=kfold)
#    print('='*30)
#    print(results.mean())
    
    kfold.get_n_splits(X, Y)
    
    for train_index, test_index in kfold.split(X, Y):
        # TRAIN DATA
        X_train, y_train = X[train_index,:,:], Y[train_index]
        # TEST DATA
        X_test, y_test = X[test_index,:,:], Y[test_index]
        
        # create model
#        model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=32, verbose=1)
        model = create_model(optimizer='nadam', activation='sigmoid')
        
        
        # Compile Model
        model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
        print(model.summary())
        
        history = model.fit(X_train, y_train, validation_split=0.3, epochs=100, batch_size=128, callbacks=[earlyStopping], class_weight=class_weight)
        print(history.history.keys())
        
        Y_pred = model.predict(X_test, verbose=1)
    
        stats = BaseStatistics(y_test, Y_pred)
        print stats
    

npath = "fasta/Bacillus_non_prom.fa"
ppath = "fasta/Bacillus_prom.fa"

#run_grid(npath, ppath)
run(npath, ppath)
