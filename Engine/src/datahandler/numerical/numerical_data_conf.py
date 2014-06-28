'''
Created on Jun 26, 2014

@author: Simon
'''
from datahandler.abstract_data_conf import AbstractDataConf
from datahandler.numerical.numerical_training_conf import NumericalTrainingConf

class NumericalDataConf(AbstractDataConf):
    '''
    Configuration to control data splitting and serialization.
    '''

    def __init__(self, lines_per_map):
        self.lines_per_map = lines_per_map

    def get_pre_proc_conf(self):
        '''
        @return: NumericalTrainingConf
        '''
        return NumericalTrainingConf(self.lines_per_map)
    
    def get_training_conf(self):
        '''
        @return: NumericalTrainingConf
        '''
        return NumericalTrainingConf(self.lines_per_map)
    
    def get_validation_conf(self):
        '''
        @return: NumericalTrainingConf
        '''
        return NumericalTrainingConf(self.lines_per_map)
