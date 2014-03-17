'''
Created on Jan 11, 2014

@author: xapharius
'''
from datahandler.AbstractDataHandler import AbstractDataHandler
from datahandler.numerical.NumericalDataProcessor import NumericalDataProcessor
from datahandler.numerical.NumericalConfiguration import NumericalConfiguration
from datahandler.numerical.numerical_pre_processor import NumericalPreProcessor

class NumericalDataHandler(AbstractDataHandler):
    '''
    DataHanlder for numerical data.
    '''

    LINES_PER_MAP = 20

    def __init__(self, nrInputDim, nrLabelDim):
        '''
        Constructor
        @param nrInputDim: number of Input Dimensions dataset has
        @param nrLabelDim: numer of Output Dimensions dataset has 
        '''
        self.nrInputDim = nrInputDim
        self.nrLabelDim = nrLabelDim

    def get_pre_processor(self):
        return NumericalPreProcessor()

    def get_data_processor(self):
        return NumericalDataProcessor(self.nrInputDim, self.nrLabelDim)

    def get_configuration(self):
        return NumericalConfiguration(self.LINES_PER_MAP)
