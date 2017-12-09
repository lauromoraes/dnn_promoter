#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:44:05 2017

@author: fnord
"""

class BaseData(object):
    def __init__(self, npath, ppath):
        self.ppath = ppath
        self.npath = npath

    def get_sequences_from_fasta(self, path):
        seqs = list()
        with open(path) as f:
            for l in f.readlines()[1::2]:
                seqs.append(l[:-1])
        return seqs
        
    def get_kmers(self, seq, k=1, step=1):
        numChunks = ((len(seq)-k)/step)+1
        mers = list()
        for i in range(0, numChunks*step-1, step):
            mers.append(seq[i:i+k])
        return mers
    
    def encode_sequences(self, seqs):
        raise NotImplementedError()
    
    def enconde_positives(self):
        return self.encode_sequences(self.get_sequences_from_fasta(self.ppath))
    
    def enconde_negatives(self):
        return self.encode_sequences(self.get_sequences_from_fasta(self.npath))
    
    def set_XY(self, negdata, posdata):
        from numpy import array
        from numpy import vstack
        Y = array([0 for x in range(negdata.shape[0])] + [1 for x in range(posdata.shape[0])])
        #Y = Y.transpose()
        X = vstack((negdata, posdata))
        assert X.shape[0]==Y.shape[0]
        self.X = X
        self.Y = Y
        return X, Y
    
    def getX(self, frame=0):
        if frame<0 or frame>3:
            return
        elif frame==0:
            return self.X
        else:
            return self.X[:, frame::3]
    
    def getY(self):
        return self.Y

    def get_negative_samples(self):
        samples = self.X[:(self.n_samples_neg)]
        assert samples.shape[0]==self.n_samples_neg
        return samples

    def get_positive_samples(self):
        samples = self.X[self.n_samples_pos:]
        assert samples.shape[0]==self.n_samples_pos
        return samples
    
    def set_data(self):
        posdata = self.enconde_positives()
        negdata = self.enconde_negatives()
        assert negdata.shape[1]==posdata.shape[1]
        self.n_samples_pos = posdata.shape[0]
        self.n_samples_neg = negdata.shape[0]
        self.sample_size = posdata.shape[1]
        self.set_XY(negdata, posdata)

    def set_kmers_encoder(self):
        from itertools import product
        from numpy import array
        from sklearn.preprocessing import LabelEncoder
        nucs = ['G','A','C','T']
        tups = list(product(nucs, repeat=self.k))
        data = array([''.join(x) for x in tups])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(data)

class SequenceProteinData(BaseData):
    STANDARD_GENETIC_CODE   =   {
        'TTT':0,    'TTC':0,    'TCT':10,    'TCC':10,
        'TAT':1,    'TAC':1,    'TGT':13,    'TGC':13,
        'TTA':2,    'TCA':10,   'TAA':20,    'TGA':20,
        'TTG':2,    'TCG':10,   'TAG':20,    'TGG':19,
        'CTT':2,    'CTC':2,    'CCT':14,    'CCC':14,
        'CAT':3,    'CAC':3,    'CGT':15,    'CGC':15,
        'CTA':2,    'CTG':2,    'CCA':14,    'CCG':14,
        'CAA':4,    'CAG':4,    'CGA':15,    'CGG':15,
        'ATT':5,    'ATC':5,    'ACT':11,    'ACC':11,
        'AAT':6,    'AAC':6,    'AGT':10,    'AGC':10,
        'ATA':5,    'ACA':11,   'AAA':16,    'AGA':15,
        'ATG':7,    'ACG':11,   'AAG':16,    'AGG':15,
        'GTT':8,    'GTC':8,    'GCT':17,    'GCC':17,
        'GAT':12,   'GAC':12,   'GGT':18,    'GGC':18,
        'GTA':8,    'GTG':8,    'GCA':17,    'GCG':17,
        'GAA':9,    'GAG':9,    'GGA':18,    'GGG':18}
    
    def __init__(self, npath, ppath):
        super(SequenceProteinData, self).__init__(npath, ppath)
        self.set_data()
    
    def transform(self, trimers):
        return [self.STANDARD_GENETIC_CODE[trimer] for trimer in trimers]
        
    def encode_sequences(self, seqs):
        from numpy import vstack
        return vstack([self.transform(self.get_kmers(seq, k=3)) for seq in seqs])

class SequenceNucsData(BaseData):
    def __init__(self, npath, ppath, k=1):
        super(SequenceNucsData, self).__init__(npath, ppath)
        self.k = k
        self.set_kmers_encoder()
        posdata = self.enconde_positives()
        negdata = self.enconde_negatives()
        assert negdata.shape[1]==posdata.shape[1]
        self.n_samples_pos = posdata.shape[0]
        self.n_samples_neg = negdata.shape[0]
        self.sample_size = posdata.shape[1]
        self.set_XY(negdata, posdata)
    
    def encode_sequences(self, seqs):
        from numpy import vstack
        return vstack([self.label_encoder.transform(self.get_kmers(seq, k=self.k)) for seq in seqs])

class SequenceSimpleData(BaseData):
    def __init__(self, npath, ppath, k=1):
        super(SequenceSimpleData, self).__init__(npath, ppath)
        self.k = k
        posdata = self.enconde_positives()
        negdata = self.enconde_negatives()
        assert negdata.shape[1]==posdata.shape[1]
        self.n_samples_pos = posdata.shape[0]
        self.n_samples_neg = negdata.shape[0]
        self.sample_size = posdata.shape[1]
        self.set_XY(negdata, posdata)
    
    def enconde_seq(self, seq):
        return [0 if x=='A' or x=='T' else 1 for x in seq]
    
    def encode_sequences(self, seqs):
        from numpy import vstack
        return vstack([self.enconde_seq(seq) for seq in seqs])

class SequenceDinucProperties(BaseData):
    
    def __init__(self, npath, ppath):
        import numpy as np
        super(SequenceDinucProperties, self).__init__(npath, ppath)
        self.k = 2
        self.set_kmers_encoder()
        # Setup tables for convert nucleotides to 2d properties matrix - multichannel signals
        self.convtable2 = np.loadtxt('dinuc_values', delimiter=',', dtype=np.float32)
        
        posdata = self.enconde_positives()
        negdata = self.enconde_negatives()
        print posdata.shape
        print negdata.shape
        assert negdata.shape[1]==posdata.shape[1]
        self.n_samples_pos = posdata.shape[0]
        self.n_samples_neg = negdata.shape[0]
        self.sample_size = posdata.shape[1]
        self.set_XY(negdata, posdata)
            
        
    def encode_seq(self, seq):
        import numpy as np
                
        convProp = lambda x, prop : np.array([ self.convtable2[prop, dinuc] for dinuc in x ])
        
        return np.vstack([ convProp(seq, i) for i in range(38) ]).reshape(38,len(seq),1)
        
#        return convertedseq.reshape(1, 38, len(seq))
        
        

    def encode_sequences(self, seqs):
        from numpy import vstack, array
        mat = vstack([self.label_encoder.transform(self.get_kmers(seq, k=self.k)) for seq in seqs])
        return array([ self.encode_seq(seq) for seq in mat ])


class SimpleHistData(BaseData):
    
    def __init__(self, npath, ppath, k=1, upto=False):
        super(SimpleHistData, self).__init__(npath, ppath)
        self.k = k
        self.upto = upto
        posdata = self.enconde_positives()
        negdata = self.enconde_negatives()
        assert negdata.shape[1]==posdata.shape[1]
        self.n_samples_pos = posdata.shape[0]
        self.n_samples_neg = negdata.shape[0]
        self.sample_size = posdata.shape[1]
        self.set_XY(negdata, posdata)
    
    def encode_sequences(self, seqs):
        from repDNA.nac import Kmer
        from numpy import vstack
        kmer = Kmer(k=self.k, upto=self.upto, normalize=True)
        return vstack(kmer.make_kmer_vec(seqs))

class DinucAutoCovarData(BaseData):
    
    def __init__(self, npath, ppath, k=1, upto=False):
        super(DinucAutoCovarData, self).__init__(npath, ppath)
        self.k = k
        self.upto = upto
        posdata = self.enconde_positives()
        negdata = self.enconde_negatives()
        assert negdata.shape[1]==posdata.shape[1]
        self.n_samples_pos = posdata.shape[0]
        self.n_samples_neg = negdata.shape[0]
        self.sample_size = posdata.shape[1]
        self.set_XY(negdata, posdata)
    
    def encode_sequences(self, seqs):
        from repDNA.ac import DAC
        from numpy import vstack
        dac = DAC(self.k)
        return vstack(dac.make_dac_vec(seqs, all_property=True))

class DinucCrossCovarData(BaseData):
    
    def __init__(self, npath, ppath, k=1, upto=False):
        super(DinucCrossCovarData, self).__init__(npath, ppath)
        self.k = k
        self.upto = upto
        posdata = self.enconde_positives()
        negdata = self.enconde_negatives()
        assert negdata.shape[1]==posdata.shape[1]
        self.n_samples_pos = posdata.shape[0]
        self.n_samples_neg = negdata.shape[0]
        self.sample_size = posdata.shape[1]
        self.set_XY(negdata, posdata)
    
    def encode_sequences(self, seqs):
        from repDNA.ac import DCC
        from numpy import vstack
        dcc = DCC(self.k)
        return vstack(dcc.make_dcc_vec(seqs, all_property=True))
