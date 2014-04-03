'''
Created on Apr 3, 2014

@author: linda
'''
from datahandler.abstract_statistics import AbstractStatistics
from datahandler.numerical.basic_numerical_stats import BasicNumericalStats

class NumericalStats(AbstractStatistics):
    '''
    Numerical statistics for labeled data. Basically consists of two 
    BasicNumericalStats for data and labels.
    '''


    def __init__(self, input_stats=None, target_stats=None):
        '''
        Creates new labeled statistics using the given statistics.
        @param input_stats: numerical input statistics
        @param target_stats: numerical target statistics
        '''
        self.input_stats = input_stats
        self.target_stats = target_stats

    def get_input_stats(self):
        return self.input_stats

    def get_target_stats(self):
        return self.target_stats

    def encode(self):
        enc_input_stats = self.input_stats.encode()
        enc_target_stats = None
        if self.target_stats is not None:
            enc_target_stats = self.target_stats.encode()
        return { 'input_stats': enc_input_stats,
                'target_stats': enc_target_stats }

    def decode(self, encoded_stats):
        self.input_stats = BasicNumericalStats().decode(encoded_stats['input_stats'])
        enc_target_stats = encoded_stats['target_stats']
        if enc_target_stats is not None:
            self.target_stats = BasicNumericalStats().decode(enc_target_stats)
        else:
            self.target_stats = None
        return self