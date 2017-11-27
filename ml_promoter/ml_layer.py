#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:02:41 2017

@author: fnord
"""

class BaseLayer(object):

    def set_layers_pipeline(self):
        raise NotImplementedError()
        
    def get_next_layer(self):
        raise NotImplementedError()
    
    def add_layer(self, L):
        self.architeture.add(L)
    
    def get_layer_parameters(self, L):
        try:
            keras
        except:
            import keras
        # Dense - fully connected
        if type(L)==keras.layers.core.Dense:
            return {
                    'num_nodes':(64,128,256)
            }
        # CNN - Convolutional

        elif type(L)==keras.layers.Conv1D:
            return {
                    'filters':(64,128,256), 
                    'kernel':(3,9), 
                    'strides':(1,3)
            }
        # RNN - LSTM
        elif type(L)==keras.layers.LSTM:
            return {
                    'states':(64,128,256), 
                    'dropout':(3,9), 
                    'recurrent_dropout':(1,3)
            }
        

class  SequenceLayers(BaseLayer):
    
    def __init__(self):
        from keras.models import Sequential
        self.architeture = Sequential()
    
    def set_layers_pipeline(self):
        
    