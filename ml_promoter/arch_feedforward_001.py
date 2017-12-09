#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 12:52:25 2017

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
    from ml_data import SimpleHistData
    from ml_data import DinucCrossCovarData
    
    global data
    global limits
    global lengths
    global shared_input_length
    
    # Join data into a single input vector
    data = join_data(
            SimpleHistData(npath, ppath, k=4, upto=True),
            DinucCrossCovarData(npath, ppath, k=3, upto=True),
    )
    
    # Get lenghts of input vectors
    lengths = get_input_lengths(data)
    
    # Define limits of data type on single vector
    limits = calc_limits(data)
    
    # Compute shared input vector length
    shared_input_length = limits[-1][-1]

def create_model(optimizer='nadam', activation='sigmoid'):
    from keras import backend as K
    from keras.models import Model
    from keras.layers import Input
    from keras.layers import Dense
    from keras.layers import concatenate
    
    # Set Shared Input
    shared_input = Input(shape=(shared_input_length, ), dtype='float32', name='single_input')
    
    # Split Inputs
    input_01 = crop_data(limits[0][0], limits[0][1])(shared_input)
    input_02 = crop_data(limits[1][0], limits[1][1])(shared_input)
    
    # Add Hidden Neurons
    dense_01 = Dense(lengths[0], activation=activation)(input_01)
    dense_02 = Dense(lengths[1], activation=activation)(input_02)
    
    # Merge branchs
    merged = concatenate([dense_01, dense_02], axis=1)
    
    # Add last neuron
#            prob = Dense(1, activation=K.sigmoid, kernel_initializer=K.random_normal)
    prob = Dense(1, activation=K.sigmoid)(merged)
    
    
    # Setup model object
    model = Model(inputs=shared_input, outputs=prob)            
    # Compile Model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
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
    
    weightTrue = 0.8
    class_weight = {0:(1-weightTrue), 1:weightTrue}
    
    # Setup all data for inputs
    setup_data(npath, ppath)
    
    # Retrieve data for model
    X, Y = get_model_data_full(data)
    
    # create model
    model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=32, verbose=1)
    
    plot_model(create_model(), show_shapes=True)
    
    # evaluate using 10-fold cross validation
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=1543)
    results = cross_val_score(model, X, Y, cv=kfold)
    print('='*30)
    print(results.mean())
    

npath = "fasta/Bacillus_non_prom.fa"
ppath = "fasta/Bacillus_prom.fa"

#run_grid(npath, ppath)
run(npath, ppath)
