'''
Created on Mar 26, 2014

@author: Simon
'''

import numpy as np
from datahandler.abstract_statistics import AbstractStatistics

class NumericalStats(AbstractStatistics):
    '''
    Aggregates numerical statistics like min, max, mean, variance.
    '''

    def __init__(self, size=0, min_val=0, max_val=0, mean_val=0, variance=0, encoded=None):
        if encoded is not None:
            self.decode(encoded)
        else:
            self.size = size
            self.set_min(min_val)
            self.set_max(max_val)
            self.set_mean(mean_val)
            self.set_variance(variance)

    def set_min(self, min_val):
        try:
            self.min = min_val.tolist()
        except AttributeError:
            self.min = min_val

    def get_min(self, as_np_array=True):
        if as_np_array:
            return np.array(self.min)
        return self.min

    def set_max(self, max_val):
        try:
            self.max = max_val.tolist()
        except AttributeError:
            self.max = max_val

    def get_max(self, as_np_array=True):
        if as_np_array:
            return np.array(self.max)
        return self.max

    def set_mean(self, mean_val):
        try:
            self.mean = mean_val.tolist()
        except AttributeError:
            self.mean = mean_val

    def get_mean(self, as_np_array=True):
        if as_np_array:
            return np.array(self.mean)
        return self.mean

    def set_variance(self, variance):
        try:
            self.variance = variance.tolist()
        except AttributeError:
            self.variance = variance

    def get_variance(self, as_np_array=True):
        if as_np_array:
            return np.array(self.variance)
        return self.variance

    def set_size(self, size):
        self.size = size

    def get_size(self):
        return self.size

    def copy_from(self, other):
        self.size = other.size
        self.min = other.min
        self.max = other.max
        self.mean = other.mean
        self.variance = other.variance

    def encode(self):
        return { 'size': self.size, 'min': self.min, 'max': self.max, 'mean': self.mean, 'variance': self.variance }

    def decode(self, encoded_stats):
        self.size = encoded_stats['size']
        self.min = encoded_stats['min']
        self.max = encoded_stats['max']
        self.mean = encoded_stats['mean']
        self.variance = encoded_stats['variance']
        return self

