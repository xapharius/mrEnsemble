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
    def encode(self, alg_instance):
        '''
        Encode an existing algorithm instance. Usually this means creating a
        representation of the instance's parameters/weights.
        :param alg_instance algorithm instance that was created by this factory
        and should be encoded
        '''
        pass
    
    @abstractmethod
    def decode(self, encoded):
        '''
        Creates a list of new algorithm instances, that represent the given encoded
        algorithm objects created through the method 'encode'.
        @param encoded: List of encoded algorithm objects
        '''
        pass

    #TODO: confAlg method necessary?
    """
    @abstractmethod
    def confAlg(self):
        '''Configure specific Algorithm'''
        pass
    """