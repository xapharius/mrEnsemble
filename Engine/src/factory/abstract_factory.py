'''
Created on Mar 22, 2015

@author: xapharius
'''
from abc import ABCMeta, abstractmethod

class AbstractFactory(object):
    '''
    Factories that instanciate managers aka Algorithm + FeatureEngineer
    '''
    __metaclass__ = ABCMeta


    def __init__(self, params):
        '''
        Constructor
        '''
        pass

    @abstractmethod
    def get_instance(self):
        pass

    @abstractmethod
    def encode(self, manager):
        pass
    
    @abstractmethod
    def decode(self, encoded):
        pass