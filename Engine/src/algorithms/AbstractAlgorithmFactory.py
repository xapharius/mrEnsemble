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
    def get_instance(self):
        '''Create an Algorithm Object
        :return: Object implementing AbstractAlgorithm
        '''
        pass
    
    @abstractmethod
    def aggregate(self, modelsArr):
        '''
        Aggregate multiple AbstractAlgorithms into one
        :param modelsArr np.array of Class implementing AbstractAlgorithm
        '''
        pass
    
    @abstractmethod
    def serialize(self, alg_instance):
        '''
        Serialize an existing algorithm instance. Usually this means creating a
        representation of the instance's parameters/weights.
        :param alg_instance algorithm instance that was created by this factory
        and should be serialized
        '''
        pass
    
    @abstractmethod
    def deserialize(self, serialized):
        '''
        Creates a new algorithm instance that represents the given serialized
        algorithm object created through the method 'serialize'.
        '''
        pass

    #TODO: confAlg method necessary?
    """
    @abstractmethod
    def confAlg(self):
        '''Configure specific Algorithm'''
        pass
    """