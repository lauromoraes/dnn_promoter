#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:12:55 2017

@author: fnord
"""

class BaseStatistics(object):
    def __init__(self, y_true, y_pred):
        from sklearn.metrics import confusion_matrix
        from numpy import array
        y_pred = array([y[0] for y in y_pred])
        y_pred_norm = y_pred > 0.5
        print 'LEN', len(y_pred)
        for i in range(len(y_pred)):
            print '{0: <10} {1: <10} {2: <10}'.format(y_true[i], y_pred_norm[i], y_pred[i])
            
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_norm).ravel()
        self.set_hits(tp, fp, tn, fn)
#        self.set_hits(tn, fn, tp, fp)
        
    def set_hits(self, tp, fp, tn, fn):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn
        self.setPrec()
        self.setSn()
        self.setSp()
        self.setAcc()
        self.setF1()
        self.setMcc()
        
    
    def setPrec(self):
        # Precision
        self.Prec = float(self.tp) / (self.tp + self.fp)
    
    def setSn(self):
        # True positive rate - Recall
        self.Sn = float(self.tp) / (self.tp + self.fn)
    
    def setSp(self):
        # True negative rate
        self.Sp = float(self.tn) / (self.tn + self.fp)
    
    def setAcc(self):
        # Accuracy
        self.Acc = float(self.tp+self.tn) / (self.tn+self.tp+self.fn+self.fp)
    
    def setF1(self):
        # F1-measure
        self.F1 = 2*float(self.Prec*self.Sn)/(self.Prec+self.Sn)
    
    def setMcc(self):
        from math import sqrt
        # Matthews correlation coefficient
        self.Mcc = float((self.tp*self.tn)-(self.fp*self.fn))/sqrt((self.tp+self.fp)*(self.tp+self.fn)*(self.tn+self.fp)*(self.tn+self.fn))
    
    def __str__(self):
        sep = '================================================================='
        line = '_________________________________________________________________'
        hits_header = '{0: <5} {1: <5} {2: <5} {3: <5}'.format('tp', 'fp', 'tn', 'fn')
        hits = '{0: <5} {1: <5} {2: <5} {3: <5}'.format(self.tp, self.fp, self.tn, self.fn)
        scores_header = '{0: <5} {1: <5} {2: <5} {3: <5} {4: <5} {5: <5}'.format('Prec', 'Sn', 'Sp', 'Acc', 'F1', 'Mcc')
        scores = '{0: <5.3} {1: <5.3} {2: <5.3} {3: <5.3} {4: <5.3} {5: <5.3}'.format(self.Prec, self.Sn, self.Sp, self.Acc, self.F1, self.Mcc)
        
        return '\n'.join([sep, hits_header, hits, line, scores_header, scores, sep])