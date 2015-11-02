'''
Created on Jan 11, 2014

@author: xapharius
'''
from datahandler.AbstractDataHandler import AbstractDataHandler
from datahandler.numerical.NumericalDataProcessor import NumericalDataProcessor
from datahandler.numerical.numerical_pre_processor import NumericalPreProcessor
from datahandler.numerical.numerical_stats import NumericalStats
from datahandler.numerical.numerical_data_conf import NumericalDataConf

class NumericalDataHandler(AbstractDataHandler):
    '''
    DataHanlder for numerical data.
    '''

    LINES_PER_MAP = 100

    def __init__(self, nrInputDim, nrLabelDim, input_scalling=None, target_scalling=None):
        '''
        Constructor
        @param nrInputDim: number of Input Dimensions dataset has
        @param nrLabelDim: numer of Output Dimensions dataset has 
        '''
        self.nrInputDim = nrInputDim
        self.nrLabelDim = nrLabelDim
        self.input_scalling = input_scalling
        self.target_scalling = target_scalling
        self.pre_proc = NumericalPreProcessor()
        self.data_proc = NumericalDataProcessor(self.nrInputDim, self.nrLabelDim, input_scalling=self.input_scalling, target_scalling=self.target_scalling)
        super(NumericalDataHandler, self).__init__()

    def get_pre_processor(self):
        return self.pre_proc

    def set_pre_processor(self, pre_proc):
        self.pre_proc = pre_proc

    def get_data_processor(self):
        return self.data_proc

    def set_data_processor(self, data_proc):
        self.data_proc = data_proc

    def get_configuration(self):
        return NumericalDataConf(self.LINES_PER_MAP)

    def get_new_statistics(self):
        return NumericalStats()
