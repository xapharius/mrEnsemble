'''
Created on Mar 12, 2014

@author: Simon
'''
from datahandler.AbstractPreProcessor import AbstractPreProcessor
import numpy as np
from datahandler.numerical.basic_numerical_stats import BasicNumericalStats
from datahandler.numerical.numerical_stats import NumericalStats

class NumericalPreProcessor(AbstractPreProcessor):
    '''
    Preprocessor implementation for numerical data sets.
    Calculates min, max, mean and variance.
    '''

    def calculate(self, data_set):
        data_stats = self._calculate_stats(data_set.inputs)
        label_stats = self._calculate_stats(data_set.targets)
        return NumericalStats(data_stats, label_stats).encode()

    def _calculate_stats(self, data):
        data_min  = np.min(data, axis=0)
        data_max  = np.max(data, axis=0)
        data_mean = np.mean(data, axis=0)
        data_var  = np.var(data, axis=0)
        return BasicNumericalStats(data.shape[0], data_min, data_max, data_mean, data_var)

    def aggregate(self, key, values):
        input_result_stats = BasicNumericalStats()
        target_result_stats = BasicNumericalStats()
        for encoded_stats in values:
            stats = NumericalStats().decode(encoded_stats)
            input_stats = stats.get_input_stats()
            target_stats = stats.get_target_stats()
            input_result_stats.combine(input_stats)
            target_result_stats.combine(target_stats)
        return NumericalStats(input_result_stats, target_result_stats).encode()
