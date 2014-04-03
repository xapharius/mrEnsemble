'''
Created on Mar 12, 2014

@author: Simon
'''
from datahandler.AbstractPreProcessor import AbstractPreProcessor
import numpy as np
from datahandler.numerical.numerical_stats import NumericalStats

class NumericalPreProcessor(AbstractPreProcessor):
    '''
    Preprocessor implementation for numerical data sets.
    Calculates min, max, mean and variance.
    '''

    def calculate(self, data_set):
        data_stats = self._calculate_stats(data_set.inputs)
        target_stats = self._calculate_stats(data_set.targets)
        return { 'data': data_stats.encode(), 'target': target_stats.encode() }

    def _calculate_stats(self, data):
        data_min  = np.min(data, axis=0)
        data_max  = np.max(data, axis=0)
        data_mean = np.mean(data, axis=0)
        data_var  = np.var(data, axis=0)
        return NumericalStats(data.shape[0], data_min, data_max, data_mean, data_var)

    def aggregate(self, key, values):
        data_result_stats = NumericalStats()
        target_result_stats = NumericalStats()
        for stats in values:
            data_stats = NumericalStats(encoded=stats['data'])
            target_stats = NumericalStats(encoded=stats['target'])
            self._aggregate_stats(data_result_stats, data_stats)
            self._aggregate_stats(target_result_stats, target_stats)
        return { 'data': data_result_stats.encode(), 'target': target_result_stats.encode() }

    def _aggregate_stats(self, result, other):
        if result.get_size() == 0:
            result.copy_from(other)
        else:
            mins = np.vstack((result.get_min(), other.get_min()))
            maxs = np.vstack((result.get_max(), other.get_max()))

            result.set_min( np.min(mins, axis=0) )
            result.set_max( np.max(maxs, axis=0) )
            result_size = result.get_size()
            other_size = other.get_size()
            total = result_size + other_size
            result.set_variance( (result.get_variance()*result_size + other.get_variance()*other_size) / total
                                 + result_size * other_size * np.power((other.get_mean()-result.get_mean()) / total, 2) )
            result.set_mean( (result.get_mean()*result_size + other.get_mean()*other_size) / total )
            result.set_size(total)
