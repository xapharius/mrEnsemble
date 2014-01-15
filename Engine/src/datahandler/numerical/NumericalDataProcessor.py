'''
Created on Jan 11, 2014

@author: xapharius
'''
from datahandler import AbstractDataProcessor
from datahandler.numerical.NumericalDataSet import NumericalDataSet

class NumericalDataProcessor(AbstractDataProcessor):
    '''
    Processor for data received as parameter from mrjob map.
    Processes the local data and returns a NumericalDataSet
    '''


    def __init__(self, rawData, nrInputDim, nrLabelDim):
        '''
        Constructor
        @param rawData: value parameter for mrjob's map, numpy.ndarray
        @param nrInputDim: number of Input Dimensions dataset has
        @param nrLabelDim: numer of Output Dimensions dataset has 
        '''
        if nrInputDim + nrLabelDim != rawData.shape[1]:
            raise Exception("Input and Label Dimensions don't add up to rawData shape")
        self.inputs = rawData[:, :nrInputDim]
        self.labels = rawData[:, nrInputDim:]
        
    def normalize_data(self):
        '''
        Get Normaization statistics through get_jobconf_value
        Normalize local data
        '''
        #TODO: get preproc stats get_jobconf_value("1DimMax") for all dims
        pass
    
    def get_data(self):
        '''
        @return: NumericalDataSet
        '''
        return NumericalDataSet(self.inputs, self.labels) 