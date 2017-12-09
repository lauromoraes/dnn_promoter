#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:41:39 2017

@author: fnord
"""

class Architecture(object):
    def __init__(self):
        pass
    
    def setup_data(self, ppath, npath):
        raise NotImplementedError()
    
    def get_architeture(self):
        raise NotImplementedError()
    
    def get_model(self):
        raise NotImplementedError()
    
    def join_data(self, *kargs):
        datalist = list()
        for data in kargs:
            datalist.append(data)
        return datalist
    
    def crop_data(start, end):
        from keras.layers.core import Lambda
        def func(x):
            return x[:, start:end]
        return Lambda(func)
    
    def get_input_lengths(self, datalist):
        return [ d.getX().shape[1] for d in datalist ]
    
    def calc_limits(self, datalist):
        input_limits = []
        start, end = 0, 0
        
        for i in range(len(datalist)):
            end = start + datalist[i].getX().shape[1]
            limit = (start, end)
            input_limits.append(limit)
            start = end
        
        return input_limits
    
    def get_model_data_full(self):
        from numpy import hstack
        Y = self.data[0].getY()
        X = hstack([x.getX() for x in self.data])
        return X, Y
        

class Arch_Feedforward_001(Architecture):
    
    def setup_data(self, npath, ppath):
        from ml_data import SimpleHistData
        from ml_data import DinucAutoCovarData
        
        # Join data into a single input vector
        data = self.join_data(
                SimpleHistData(npath, ppath, k=4),
                DinucAutoCovarData(npath, ppath),
        )
        
        # Get lenghts of input vectors
        lengths = self.get_input_lengths(data)
        
        # Define limits of data type on single vector
        limits = self.calc_limits(data)
        
        self.single_input_length = limits[-1][-1]
        self.data = data
        self.limits = limits
        self.lengths = lengths

    
    def get_model(self):
        from keras.wrappers.scikit_learn import KerasClassifier
        
        def create_model(optimizer='nadam', activation='sigmoid'):
            from keras import backend as K
            from keras.models import Model
            from keras.layers import Input
            from keras.layers import Dense
            from keras.layers import concatenate
            
            # Set Shared Input
            shared_input = Input(shape=(self.single_input_length, ), dtype='float32', name='single_input')
            
            # Split Inputs
            input_01 = self.crop_data(self.limits[0][0], self.limits[0][1])
            input_02 = self.crop_data(self.limits[1][0], self.limits[1][1])
            
            # Add Hidden Neurons
            dense_01 = Dense(self.lengths[0], activation=activation)(input_01)
            dense_02 = Dense(self.lengths[1], activation=activation)(input_02)
            
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
        
        model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
        
        return model
    
        def run_grid(self, weightTrue=0.75):
        from sklearn.model_selection import GridSearchCV
        
        # Retrieve data
        X, Y = self.get_model_data_full()
        
        model = self.get_model()
        
        param_grid = dict(
                activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
                )

        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
        
#        grid_result = grid.fit(X, Y, class_weigths={0:(1-weightTrue), 1:weightTrue}, validation_split=0.1)
        grid_result = grid.fit(X, Y)
        
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))