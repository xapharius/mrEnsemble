'''
Created on Jan 11, 2014

@author: xapharius
'''
from datahandler.AbstractDataProcessor import AbstractDataProcessor
from datahandler.numerical.NumericalDataSet import NumericalDataSet
import numpy as np 

class NumericalDataProcessor(AbstractDataProcessor):
    '''
    Processor for data received as parameter from mrjob map.
    Processes the local data and returns a NumericalDataSet
    '''
    
    NORMALIZE   = 'normalize'
    STANDARDIZE = 'standardize'
    UNIT_LENGTH = 'unit_length'


    def __init__(self, nr_input_dim, nr_target_dim, input_scalling=None, target_scalling=None):
        '''
        Constructor
        @param nrInputDim: number of Input Dimensions dataset has
        @param nrLabelDim: numer of Output Dimensions dataset has 
        '''
        self.nr_input_dim = nr_input_dim
        self.nr_target_dim = nr_target_dim
        self.input_scalling = input_scalling
        self.target_scalling = target_scalling

    def normalize_data(self, stats):
        '''
        Normalize local data
        '''
        input_stats = stats.get_input_stats()
        target_stats = stats.get_target_stats()

        try:
            # feature scalling
            self.inputs = self._scale(self.inputs, input_stats, self.input_scalling)
            self.targets = self._scale(self.targets, target_stats, self.target_scalling)
        except AttributeError:
            raise Exception("'set_data' has to be called before calling 'normalize_data'!")

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

        super(NumericalDataProcessor, self).set_data(raw_data)

    def get_data_set(self):
        '''
        @return: NumericalDataSet
        '''
        return NumericalDataSet(self.inputs, self.targets)
