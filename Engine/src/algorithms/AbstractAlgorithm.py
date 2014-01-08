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
  
    #TODO: Dataset type as argument    
    @abstractmethod
    def train(self, dataSet):
        pass
    
    #TODO: Dataset type as argument    
    @abstractmethod
    def predict(self, dataSet):
        pass
    
    @abstractmethod
    def set_params(self, parameters):
        pass