'''
Created on Jan 11, 2014

@author: xapharius
'''
from datahandler.AbstractDataProcessor import AbstractDataProcessor
from datahandler.numerical.NumericalDataSet import NumericalDataSet
import numpy as np 
from utils import logging
from datahandler.numerical.numerical_stats import NumericalStats

class PenDigitsDataProcessor(AbstractDataProcessor):
    '''
    Processor for the pen digits data set.
    '''
    
    NORMALIZE   = 'normalize'
    STANDARDIZE = 'standardize'
    UNIT_LENGTH = 'unit_length'


    def __init__(self, nr_input_dim, input_scalling=None):
        '''
        Constructor
        @param nrInputDim: number of Input Dimensions dataset has
        @param nrLabelDim: numer of Output Dimensions dataset has 
        '''
        self.nr_input_dim = nr_input_dim
        self.input_scalling = input_scalling
        self.nr_target_dim = 1

    def normalize_data(self, stats):
        '''
        Normalize local data
        '''
        stats = NumericalStats().decode(stats)
        input_stats = stats.get_input_stats()

        try:
            # feature scalling
            self.inputs = self._scale(self.inputs, input_stats, self.input_scalling)
            self.targets = self.convert_targets(self.targets)
        except AttributeError:
            raise Exception("'set_data' has to be called before calling 'normalize_data'!")

    def convert_targets(self, targets):
        result = np.zeros( (len(targets), 10) )
        for i in range(len(targets)):
            result[i, targets[i,0]] = 1
        return np.array(result)

    def _scale(self, data, stats, scalling):
        if scalling is None:
            return data
        elif scalling == self.STANDARDIZE:
            result = (data - stats.get_mean()) / stats.get_variance()
        elif scalling == self.NORMALIZE:
            result = (data - stats.get_min()) / (stats.get_max() - stats.get_min())
        elif scalling == self.UNIT_LENGTH:
            result = np.array(map(lambda row: row / np.linalg.norm(row), data))
        return result

    def set_data(self, raw_data):
        '''
        @param rawData: value parameter for mrjob's map, numpy.ndarray
        '''
        if self.nr_input_dim + self.nr_target_dim != raw_data.shape[1]:
            raise Exception("Input and Label Dimensions don't add up to rawData shape")
        self.inputs = raw_data[:, :self.nr_input_dim]
        self.targets = raw_data[:, self.nr_input_dim:]

        super(PenDigitsDataProcessor, self).set_data(raw_data)

    def get_data_set(self):
        '''
        @return: NumericalDataSet
        '''
        return NumericalDataSet(self.inputs, self.targets)
