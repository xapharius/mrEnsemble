'''
Created on Mar 26, 2014

@author: Simon
'''

import numpy as np
from datahandler.abstract_statistics import AbstractStatistics

class BasicNumericalStats(AbstractStatistics):
    '''
    Aggregates numerical statistics like min, max, mean, variance.
    '''

    def __init__(self, size=0, min_val=0, max_val=0, mean_val=0, variance=0):
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
        '''
        Copies all values from the given BasicNumericalStats instance.
        @param other: BasicNumericalStats instance to copy from
        @return: This instance
        '''
        self.size = other.size
        self.min = other.min
        self.max = other.max
        self.mean = other.mean
        self.variance = other.variance
        return self

    def combine(self, other):
        '''
        Combines the giBasicNumericalStatstats instance and this instance, where this
        instance holds the result.
        @param other: AnotBasicNumericalStatstats instance
        @return: This instance
        '''
        if self.get_size() == 0:
            self.copy_from(other)
        else:
            mins = np.vstack((self.get_min(), other.get_min()))
            maxs = np.vstack((self.get_max(), other.get_max()))

            self.set_min( np.min(mins, axis=0) )
            self.set_max( np.max(maxs, axis=0) )
            self_size = self.get_size()
            other_size = other.get_size()
            total = self_size + other_size
            self.set_variance( (self.get_variance()*self_size + other.get_variance()*other_size) / total
                                 + self_size * other_size * np.power((other.get_mean()-self.get_mean()) / total, 2) )
            self.set_mean( (self.get_mean()*self_size + other.get_mean()*other_size) / total )
            self.set_size(total)
        return self

    def encode(self):
        return { 'size': self.size, 'min': self.min, 'max': self.max, 'mean': self.mean, 'variance': self.variance }

    def decode(self, encoded_stats):
        self.size = encoded_stats['size']
        self.min = encoded_stats['min']
        self.max = encoded_stats['max']
        self.mean = encoded_stats['mean']
        self.variance = encoded_stats['variance']
        return self

