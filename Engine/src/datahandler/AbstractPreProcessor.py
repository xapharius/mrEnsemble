'''
Created on Jan 8, 2014

@author: xapharius
'''

from abc import ABCMeta, abstractmethod
from mrjob.job import MRJob
import mrjob

class AbstractPreProcessor(object):
    '''
    Abstract class for a Preprocessor.
    Specifies which methods a Preprocessor of a Data Class should implement.
    The Preprocessor gets the necessary statistics for the DataProcessor through m/r on the dataset, 
    before the actual m/r of the engine.
    The statistics must then be passed to the DataProcessor through the engines Job Config
    '''
    __metaclass__ = ABCMeta
    
    def __init__(self, params):
        '''
        Constructor
        '''
        pass  
        
    #TODO: map/reduce evoke here or data dependent?    
    @abstractmethod
    def get_statistics(self, dataSource):
        '''
        Creates dictionary with statistics of the dataset
        Dictionary is constructed through map/reduce as the dataset is stored in HDFS
        Statistics are specific for each Dataclass (e.g images, timeseries etc)
        @param dataSource: location on HDFS of data that wants to be analysed 
        @return: Dictionary of collected statistics
        '''
        pass
    
    @abstractmethod
    def mapper(self):
        '''
        Map/Reduce map to get statistics
        '''
        pass
    
    @abstractmethod
    def reducer(self):
        '''
        Map/Reduce reduce to get statistics
        '''
        pass
            
        