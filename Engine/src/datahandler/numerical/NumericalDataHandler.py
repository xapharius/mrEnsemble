'''
Created on Jan 11, 2014

@author: xapharius
'''
from datahandler import AbstractDataHandler
from datahandler.numerical.NumericalDataProcessor import NumericalDataProcessor
from datahandler.numerical.NumericalConfiguration import NumericalConfiguration

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
        
    #TODO: implement get_PreProcessor
    def get_PreProcessor(self):
        pass
    
    def get_DataProcessor(self, rawData):
        return NumericalDataProcessor(rawData, self.nrInputDim, self.nrLabelDim)
    
    def get_configuration(self):
        return NumericalConfiguration(self.LINES_PER_MAP)
        