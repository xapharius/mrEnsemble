'''
Created on Mar 22, 2015

@author: xapharius
'''

from abc import ABCMeta, abstractmethod

class AbstractEnsemble(object):
    '''
    classdocs
    '''
    __metaclass__ = ABCMeta

    def __init__(self, params):
        '''
        Constructor
        '''
    
    @abstractmethod
    def predict(self, raw_data):
        '''
        @param raw_data: np.ndarray "raw" since each model has it's own feature selector
        '''
        pass