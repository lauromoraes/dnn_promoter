#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:52:49 2017

@author: fnord
"""

class BaseExperiment(object):
    
    errors = (
            'mean_squared_error', # 0
            'mean_absolute_error', # 1
            'mean_absolute_percentage_error', # 2
            'mean_squared_logarithmic_error', # 3
            'squared_hinge', # 4
            'hinge', # 5
            'categorical_hinge', # 6
            'logcosh', # 7
            'categorical_crossentropy', # 8
            'sparse_categorical_crossentropy', # 9
            'binary_crossentropy', # 10
            'kullback_leibler_divergence', # 11
            'poisson', # 12
            'cosine_proximity' # 13
            )
    
    def __init__(self, model, num_folds=3):
        from keras.models import Sequential
        self.model = model
        self.num_folds = num_folds
        self.setSeed()
        self.setKFolds()
        self.setCallbacks()
        
        self.error_metric = self.errors[10]
        self.batch_size = 64
        self.epochs = 300
        self.optimizer = 'adam'
        
        self.validation_split = 0.1
        self.class_weigths = {
                0: 0.8,
                1: 0.2
        }
    
    def setSeed(self, seed=13):
        self.seed = seed
    
    def setOptimizer(self, opt='adam'):
        self.optimizer = opt
    
    def setKFolds(self):
        from sklearn.model_selection import StratifiedKFold
        self.folds = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
    
    def setCallbacks(self):
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', epsilon=0.001, cooldown=0, min_lr=0)
        
        self.callbacks = []
        self.callbacks.append( earlyStopping )
#        self.callbacks.append( reduceLR )
    
    def print_history(self, history):
        import matplotlib.pyplot as plt
        
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
    
    def run(self):
        from ml_statistics import BaseStatistics
        
        X = self.model.get_model_X_rows()
        Y = self.model.get_model_Y_rows()
        self.folds.get_n_splits(X, Y)
        
        for train_index, test_index in self.folds.split(X, Y):
            
            # TRAIN instances
            X_train_inputs, y_train = self.model.get_model_data(train_index)
            
            # TEST instances
            X_test_inputs, y_test = self.model.get_model_data(test_index)            
            
            
            self.model.model.compile(loss=self.error_metric, optimizer=self.optimizer, metrics=['accuracy'])
            print(self.model.model.summary())
            
            history = self.model.model.fit(X_train_inputs, y_train, validation_split=self.validation_split, epochs=self.epochs, batch_size=self.batch_size, callbacks=self.callbacks, class_weight=self.class_weigths, verbose=0)
            print(history.history.keys())
            
            self.print_history(history)
            
            Y_pred = self.model.model.predict(X_test_inputs)
            
            stats = BaseStatistics(y_test, Y_pred)
            print stats
            print '='*30
    
    def create_model(self):
        
    
    def run(self, architecture, data):
        pass
    
    def run_grid(self):
        from ml_statistics import BaseStatistics
        from keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.model_selection import GridSearchCV
        
        # Get Data
        X, Y = self.model.get_model_data_full()
        
        # Put model on wrapper
        def create_model(optimizer='Nadam', error_metric='binary_crossentropy'):
            self.model.model.compile(loss=error_metric, optimizer=optimizer, metrics=['accuracy'])
            print(self.model.model.summary())
            return self.model.model
        
        # create model
        mymodel = KerasClassifier(build_fn=create_model, epochs=100, batch_size=64, verbose=0)
        
        param_grid = dict()
#        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
        grid = GridSearchCV(estimator=mymodel, param_grid=param_grid, n_jobs=1, cv=2)
        grid_result = grid.fit(X, Y)
        
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        
        