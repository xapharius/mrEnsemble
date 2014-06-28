'''
Created on Jan 15, 2014

@author: Simon
'''
from abc import ABCMeta, abstractmethod

class AbstractDataConf(object):
    '''
    Configuration to control data splitting and serialization.
    '''

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_pre_proc_conf(self):
        '''
        @return: AbstractJobConf for the pre-processing job
        '''
        pass
    
    @abstractmethod
    def get_training_conf(self):
        '''
        @return: AbstractJobConf for the training job
        '''
        pass
    
    @abstractmethod
    def get_validation_conf(self):
        '''
        @return: AbstractJobConf for the validation job
        '''
        pass
