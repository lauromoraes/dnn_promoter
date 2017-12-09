#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 23:03:44 2017

@author: Lauro Moraes
"""

import numpy as np

class BaseModel(object):
    def __init__(self, npath, ppath):
        self.data_list = self.setup_data(npath, ppath)
        self.model = self.setup_model_architecture()
    
    def setup_data(self, npath, ppath):
        raise NotImplementedError()
    
    def get_num_data_models(self):
        return len(self.data_list)
    
    def setup_model_architecture(self):
        raise NotImplementedError()
    
    def get_model_X_rows(self):
        return self.data_list[0].getX()
    
    def get_model_Y_rows(self):
        return self.data_list[0].getY()
        
    def get_model_X(self, index=None):
        if index is not None:
            return [ d.getX()[index,:] for d in self.data_list ]
        else:
            return [ d.getX()[:,:] for d in self.data_list ]
    
    def get_model_Y(self, index):
        return self.data_list[0].getY()[index]
    
    def get_model_data(self, index=None):
        return self.get_model_X(index), self.get_model_Y(index)
    
    def get_model_data_full(self):
        Y = self.get_model_Y_rows()
        X = np.hstack([x.getX() for x in self.data_list])
        return X, Y
    
    def get_model(self):
        return self.model

class Model_Hist_FF1(BaseModel):
    
    def __init__(self, npath, ppath):
        super(Model_Hist_Conv1, self).__init__(npath, ppath)
    
    def setup_data(self, npath, ppath):
        from ml_data import SimpleHistData
        from ml_data import DinucAutoCovarData
        
        self.data_list = []
        
        self.data_list.append( SimpleHistData(npath, ppath, k=4) )
        self.data_list.append( DinucAutoCovarData(npath, ppath) )
        
        return self.data_list
        
    def setup_model_architecture(self):
        from keras.models import Model
        from keras.layers import Input, Dense, merge, concatenate
        
        input_shapes = [ x.getX().shape[1] for x in self.data_list ]
        
        first_input = Input(shape=(input_shapes[0], ))
        first_dense = Dense(1, )(first_input)
        
        second_input = Input(shape=(input_shapes[1], ))
        second_dense = Dense(1, )(second_input)
        
        merge_one = concatenate([first_dense, second_dense])
        fully = Dense(1, )(merge_one)
        
        model = Model(inputs=[first_input, second_input], outputs=fully)
        
        self.model = model
        
        return self.model
    
class Model_Hist_Conv1(BaseModel):
    
    def __init__(self, npath, ppath):
        super(Model_Hist_Conv1, self).__init__(npath, ppath)
    
    def setup_data(self, npath, ppath):
        from ml_data import SimpleHistData
        from ml_data import DinucAutoCovarData
        
        self.data_list = []
        
        self.data_list.append( SimpleHistData(npath, ppath, k=4) )
        self.data_list.append( DinucAutoCovarData(npath, ppath, k=4) )
        
        return self.data_list
        
    def setup_model_architecture(self):
        from keras.models import Model
        from keras.layers import Input, Dense, merge, concatenate, Conv1D, Flatten, MaxPooling1D, Reshape, Dropout
        from keras.layers.core import Lambda
        from keras import backend as K
        
        def calc_limits(l):
            input_limits = list()
            start = 0
            end = 0
            for i in range(len(l)):
                end = start + l[i].getX().shape[1]
                limit = (start, end)
                input_limits.append(limit)
                start = end
            return input_limits
        
        def crop(start, end):
            def func(x):
                return x[:, start:end]
            return Lambda(func)
        
        input_limits = calc_limits(self.data_list)
        
        input_shapes = [ x.getX().shape[1] for x in self.data_list ]
        
        last_index = input_limits[-1][-1]
        
        # Get input - full
        initial_input = Input(shape=(last_index, ), dtype='float32', name='initial_input')
        
        # Split inputs
        left_input = crop(input_limits[0][0], input_limits[0][1])(initial_input)
        right_input = crop(input_limits[1][0], input_limits[1][1])(initial_input)
        
        left_dense = Dense(input_shapes[0], kernel_initializer=K.random_uniform, activation=K.sigmoid)(left_input)
        right_dense = Dense(input_shapes[1], kernel_initializer=K.random_uniform, activation=K.sigmoid)(right_input)
        
        merged = concatenate([left_dense, right_dense], axis=1)
        
        dropped = Dropout(0.1)(merged)
#        hidden1 = Dense(2, activation=K.sigmoid)(merged)
        prob = Dense(1, kernel_initializer=K.random_uniform, activation=K.sigmoid)(dropped)
        
        model = Model(inputs=initial_input, outputs=prob)
        
        
        
#        print(self.data_list[0].getX())
        
#        first_input = Input(shape=(input_shapes[0], ), dtype='float32', name='first_input')
#        first_reshape = Reshape((input_shapes[0], 1))(first_input)
#        first_conv = Conv1D(100, kernel_size=5, strides=1, activation=K.relu, name='conv1')(first_reshape)
#        first_pool = MaxPooling1D(pool_size=2)(first_conv)
#        first_flat = Flatten()(first_pool)
#        
#        second_input = Input(shape=(input_shapes[1], ), dtype='float32', name='second_input')
#        second_reshape = Reshape((input_shapes[1], 1))(second_input)
#        second_conv = Conv1D(100, kernel_size=5, strides=1, activation=K.relu, name='conv2')(second_reshape)
#        second_pool = MaxPooling1D(pool_size=2)(second_conv)
#        second_flat = Flatten()(second_pool)
#        
#        merged = concatenate([first_flat, second_flat], axis=-1)
#        
#        hidden1 = Dense(64, activation=K.sigmoid)(merged)
#        prob = Dense(1, activation=K.softplus)(hidden1)
#        
#        model = Model(inputs=[first_input, second_input], outputs=prob)
#        
        self.model = model
        
        return self.model
    
class Model_Hist_Conv2(BaseModel):
    
    def __init__(self, npath, ppath):
        super(Model_Hist_Conv1, self).__init__(npath, ppath)
    
    def setup_data(self, npath, ppath):
        from ml_data import SimpleHistData
        from ml_data import DinucAutoCovarData
        
        self.data_list = []
        
        self.data_list.append( SimpleHistData(npath, ppath, k=4) )
        self.data_list.append( DinucAutoCovarData(npath, ppath) )
        
        return self.data_list
        
    def setup_model_architecture(self):
        from keras.models import Model
        from keras.layers import Input, Dense, merge, concatenate
        from keras import backend as K
        
        input_shapes = [ x.getX().shape[1] for x in self.data_list ]
        
        
        
        first_input = Input(shape=(input_shapes[0], ))
        first_dense = Dense(1, activation=K.elu)(first_input)
        
        second_input = Input(shape=(input_shapes[1], ))
        second_dense = Dense(1, activation=K.elu)(second_input)
        
        merge_one = concatenate([first_dense, second_dense])
        fully = Dense(1, )(merge_one)
        
        model = Model(inputs=[first_input, second_input], outputs=fully)
        
        self.model = model
        
        return self.model    
        
