'''
Created on Jan 8, 2014

@author: xapharius
'''

from abc import ABCMeta, abstractmethod

class AbstractDataHandler(object):
    '''
    Abstract class for a DataHandler. 
    Specifies which methods a DataHandler should implement. The DataHandler is responsible for a certain Data class
    (e.g images, timeseries etc), so the necessary pre- and dataprocessors belong to the same class and work together. 
    '''
    __metaclass__ = ABCMeta
    
    def __init__(self, params):
        '''
        Constructor
        '''
        pass
        
    @abstractmethod
    def get_PreProcessor(self):
        '''
        @return: Creates a PreProcessor for DataHandler's Data Class
        '''
        pass
    
    @abstractmethod
    def get_DataProcessor(self, rawData):
        '''
        @return: Creates a DataProcessor for DataHandler's Data Class and associates it rawData.
        '''
        pass
    
    @abstractmethod
    def get_configuration(self):
        '''
        @return: Configuration specifying MrJob protocols for data exchange.
        @rtype: AbstractConfiguration
        '''
        pass