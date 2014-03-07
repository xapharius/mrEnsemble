'''
Created on Jan 8, 2014

@author: xapharius
'''

from abc import ABCMeta, abstractmethod

class AbstractDataSet(object):
    '''
    Abstract class for DataSet
    DataSet is the processed rawData got in the engine's map step.
    It has the necessary format for the learning algorithms to operate.
    '''
    __metaclass__ = ABCMeta

    def __init__(self, params):
        '''
        Constructor
        '''
        pass
    
    @abstractmethod
    def get_observation(self, nr):
        '''
        Get Observation from input matrix as tuple of input and target
        '''
        pass
    
    @abstractmethod
    def gen_observations(self):
        '''
        Iterate over all observations using a generator (e.g for online training)
        '''
        pass