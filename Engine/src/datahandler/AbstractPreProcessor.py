'''
Created on Jan 8, 2014

@author: xapharius
'''

from abc import ABCMeta, abstractmethod

class AbstractPreProcessor(object):
    '''
    Abstract class for a Preprocessor.
    Specifies which methods a Preprocessor of a Data Class should implement.
    The Preprocessor gets the necessary statistics for the DataProcessor through m/r on the dataset, 
    before the actual m/r of the engine.
    The statistics must then be passed to the DataProcessor through the engines Job Config
    '''
    __metaclass__ = ABCMeta

    def __init__(self):
        '''
        Constructor
        '''
        pass  

    @abstractmethod
    def calculate(self, data_set):
        '''
        Actual pre-processing happens here. I.e. should determine necessary 
        information from the given values, like max, min, avg, ...
        This is basically the Map step of the M/R pre-processing.
        '''
        pass

    @abstractmethod
    def aggregate(self, key, values):
        '''
        Results from different pre-processor's 'calculate' should be merged here
        to give the overall result of the pre-processing.
        This is basically the Reduce step of the M/R pre-processing.
        '''
        pass
