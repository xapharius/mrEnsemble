'''
Created on Dec 4, 2013

@author: xapharius
'''

from abc import ABCMeta, abstractmethod

class AbstractAlgorithm(object):
    '''
    Abstract Class of an algorithm. 
    Specifies the core functionalities an Algorithm should implement.
    '''
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def train(self, dataSet):
        '''
        @param dataSet: AbstractDataSet
        '''
        
        pass
    
    @abstractmethod
    def predict(self, dataSet):
        '''
        @param dataSet: AbstractDataSet
        '''
    
    @abstractmethod
    def set_params(self, parameters):
        '''
        @param parameters must match in shape with the values given to Factory. Model can't change parameter shape.
        '''
        pass