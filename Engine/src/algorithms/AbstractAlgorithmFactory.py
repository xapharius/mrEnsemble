'''
Created on Dec 4, 2013

@author: xapharius
'''

from abc import ABCMeta, abstractmethod

class AbstractAlgorithmFactory(object):
    '''
    Abstract Class of an algorithm factory. 
    Specifies the core functionalities the factory should implement.
    '''
    __metaclass__ = ABCMeta
        
    @abstractmethod
    def instanciate(self):
        '''Create an Algorithm Object
        :return: Object implementing AbstractAlgorithm
        '''
        pass
    
    @abstractmethod
    def aggregate(self):
        '''Aggregate multiple AbstractAlgorithms into one'''
        pass
    
    @abstractmethod
    def confAlg(self):
        '''Configure specific Algorithm'''
        pass