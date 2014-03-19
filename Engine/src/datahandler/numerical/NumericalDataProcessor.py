'''
Created on Jan 11, 2014

@author: xapharius
'''
from datahandler.AbstractDataProcessor import AbstractDataProcessor
from datahandler.numerical.NumericalDataSet import NumericalDataSet
from datahandler.numerical.numerical_pre_processor import NumericalPreProcessor

class NumericalDataProcessor(AbstractDataProcessor):
    '''
    Processor for data received as parameter from mrjob map.
    Processes the local data and returns a NumericalDataSet
    '''


    def __init__(self, nr_input_dim, nr_label_dim):
        '''
        Constructor
        @param nrInputDim: number of Input Dimensions dataset has
        @param nrLabelDim: numer of Output Dimensions dataset has 
        '''
        self.nr_input_dim = nr_input_dim
        self.nr_label_dim = nr_label_dim

    def normalize_data(self, stats):
        '''
        Get Normaization statistics through get_jobconf_value
        Normalize local data
        '''
        try:
            self.inputs = (self.inputs - stats[NumericalPreProcessor.DATA][NumericalPreProcessor.MEAN]) / stats[NumericalPreProcessor.DATA][NumericalPreProcessor.VAR]
            self.labels = (self.labels - stats[NumericalPreProcessor.LABEL][NumericalPreProcessor.MEAN]) / stats[NumericalPreProcessor.LABEL][NumericalPreProcessor.VAR]
        except AttributeError:
            raise Exception("'set_data' has to be called before calling 'normalize_data'!")

    def set_data(self, raw_data):
        '''
        @param rawData: value parameter for mrjob's map, numpy.ndarray
        '''
        if self.nr_input_dim + self.nr_label_dim != raw_data.shape[1]:
            raise Exception("Input and Label Dimensions don't add up to rawData shape")
        self.inputs = raw_data[:, :self.nr_input_dim]
        self.labels = raw_data[:, self.nr_input_dim:]

        super(NumericalDataProcessor, self).set_data(raw_data)

    def get_data_set(self):
        '''
        @return: NumericalDataSet
        '''
        return NumericalDataSet(self.inputs, self.labels)
