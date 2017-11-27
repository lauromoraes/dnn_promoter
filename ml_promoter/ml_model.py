#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 23:03:44 2017

@author: Lauro Moraes
"""

import numpy as np

class BaseModel(object):
    def __init__(self):
        pass
    
    def setup_data(self, *kargs):
        self.data = []
        for d in kargs:
            self.data.append(d)
    
    def setup_model(self):
        from keras.models import Sequential
        self.model = Sequential()
    
    def get_model_X(self, index):
        X = []
        for d in self.data:
            X.append(d.getX()[index,:])
        return X
    
    def get_model_Y(self, index):
        return self.data[0].getY()
    
    def get_model_data(self, index):
        return self.get_model_X, self.get_model_Y
    
    def get_model(self):
        return self.model
