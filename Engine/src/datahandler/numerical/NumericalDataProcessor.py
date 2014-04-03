'''
Created on Jan 11, 2014

@author: xapharius
'''
from datahandler.AbstractDataProcessor import AbstractDataProcessor
from datahandler.numerical.NumericalDataSet import NumericalDataSet

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
            # input scalling
            if self.input_scalling is None:
                pass
            elif self.input_scalling == self.STANDARDIZE:
                self.inputs = (self.inputs - input_stats.get_mean()) / input_stats.get_variance()
            elif self.input_scalling == self.NORMALIZE:
                # TODO:
                pass
            elif self.input_scalling == self.UNIT_LENGTH:
                # TODO:
                pass

            # target scalling
            if  self.target_scalling is None:
                pass
            elif self.target_scalling == self.STANDARDIZE:
                self.targets = (self.targets - target_stats.get_mean()) / target_stats.get_variance()
            elif self.target_scalling == self.NORMALIZE:
                # TODO:
                pass
            elif self.target_scalling == self.UNIT_LENGTH:
                # TODO:
                pass

        except AttributeError:
            raise Exception("'set_data' has to be called before calling 'normalize_data'!")


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
