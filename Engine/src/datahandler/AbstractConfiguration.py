'''
Created on Jan 15, 2014

@author: Simon
'''
from abc import ABCMeta, abstractmethod

class AbstractConfiguration(object):
    '''
    Configuration to control data splitting and serialization.
    '''

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_input_protocol(self):
        '''
        @return: MrJob input protocol for data serialization
        '''
        pass
    
    @abstractmethod
    def get_internal_protocol(self):
        '''
        @return: MrJob internal protocol for data serialization
        '''
        pass
    
    @abstractmethod
    def get_output_protocol(self):
        '''
        @return: MrJob ouput protocol for data serialization
        '''
        pass
    
    @abstractmethod
    def get_hadoop_input_format(self):
        '''
        @return: Java input format implementation that should be used.
        '''
        pass
    
    @abstractmethod
    def get_job_conf(self):
        '''
        @return: Dictionary containing arbitrary job related configuration
        '''
        pass