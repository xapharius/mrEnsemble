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
        