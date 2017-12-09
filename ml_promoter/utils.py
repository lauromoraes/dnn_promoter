#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 12:49:50 2017

@author: fnord
"""

def join_data(*kargs):
    datalist = list()
    for data in kargs:
        datalist.append(data)
    return datalist

def crop_data(start, end):
    from keras.layers.core import Lambda
    def func(x):
        return x[:, start:end]
    return Lambda(func)

def get_input_lengths(datalist):
    return [ d.getX().shape[-1] for d in datalist ]

def calc_limits(datalist):
    input_limits = []
    start, end = 0, 0
    
    for i in range(len(datalist)):
        end = start + datalist[i].getX().shape[-1]
        limit = (start, end)
        input_limits.append(limit)
        start = end
    
    return input_limits

def get_model_data_full(datalist):
    from numpy import hstack
    Y = datalist[0].getY()
    X = hstack([x.getX() for x in datalist])
    X = x.getX()
    return X, Y