#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:52:49 2017

@author: fnord
"""

class BaseExperiment(object):
    
    def __init__(self, model, options, data, num_folds=5):
        from keras.models import Sequential
        self.model = Sequential()
        self.options = options
        self.data = data
        self.num_folds = num_folds
        self.setSeed()
        self.setKFolds()
    
    def setSeed(self, seed=13):
        self.seed = seed
    
    def setOptimizer(self, opt='adam'):
        self.optimizer = opt
    
    def setKFolds(self):
        from sklearn.model_selection import StratifiedKFold
        self.folds = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
    
    def run(self):
        X = self.data.getX()
        Y = self.data.getY()
        self.kfolds.get_n_splits(X, Y)
        for train_index, test_index in self.kfolds.split(X, Y):
            
            # TRAIN instances
            X_train = X[train_index,:]
            y_train = Y[train_index,:]
            # TEST instances
            X_test = X[test_index,:]
            y_test = Y[test_index,:]


class ArchitetureExperiment(BaseExperiment):
    def __init__(self, model, options, data, num_folds=5):
        super(ArchitetureExperiment, self).__init__(model, options, data, num_folds)
        pass
        
        