'''
Created on Jan 8, 2014

@author: xapharius
'''

from abc import ABCMeta, abstractmethod

class AbstractDataProcessor(object):
    '''
    Abstract class for a Dataprocessor.
    Specifies which methods a Dataprocessor of a Data Class should implement.
    The Dataprocessor is responsible of processing the data received in the map phase of the engine, in order
    for the model to learn.
    '''
    __metaclass__ = ABCMeta
    
    def __init__(self):
        '''
        Constructor
        '''

    def set_data(self, raw_data):
        self.raw_data = raw_data

    @abstractmethod
    def normalize_data(self, stats):
        '''
        Normalizes Local Data using the statistics from the preprocessor.
        '''
        pass

    @abstractmethod
    def get_data_set(self):
        '''
        Package and return processed data
        @return: Dataclass specific DataSet
        '''
        pass

