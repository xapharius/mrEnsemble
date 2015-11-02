'''
Created on Jan 8, 2014

@author: xapharius
'''

from abc import ABCMeta, abstractmethod
from engine.constants.internal import PHASE_PRE_PROC

class AbstractDataHandler(object):
    '''
    Abstract class for a DataHandler. 
    Specifies which methods a DataHandler should implement. The DataHandler is responsible for a certain Data class
    (e.g images, timeseries etc), so the necessary pre- and dataprocessors belong to the same class and work together. 
    '''
    __metaclass__ = ABCMeta

    def __init__(self):
        '''
        Constructor
        '''
        self.phase = PHASE_PRE_PROC
    
    def set_phase(self, new_phase):
        self.phase = new_phase
    
    def get_phase(self):
        return self.phase

    @abstractmethod
    def get_pre_processor(self):
        '''
        @return: Creates a PreProcessor for DataHandler's Data Class
        '''
        pass

    @abstractmethod
    def get_data_processor(self):
        '''
        @return: Creates a DataProcessor for DataHandler's Data Class.
        '''
        pass

    @abstractmethod
    def get_configuration(self):
        '''
        @return: Configuration that specifies Hadoop InputFormats and MrJob 
        protocols for pre-processing, training and validation.
        @rtype: AbstractDataConf
        '''
        pass

    @abstractmethod
    def get_new_statistics(self):
        '''
        @return: A new statistics instance implementing AbstractStatistics.
        @rtype: AbstractStatistics
        '''
        pass

    def set_statistics(self, stats):
        '''
        Set results of pre-processing.
        @param stats: Statistics calculated in pre-processing
        '''
        self.stats = stats
    
    def get_statistics(self):
        '''
        @return: Statistics that were calculated in the pre-processing step.
        @rtype: Depends on pre-processing job implementation
        '''
        return self.stats
