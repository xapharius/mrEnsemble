'''
Created on Mar 23, 2015

@author: xapharius
'''
from factory.abstract_factory import AbstractFactory
from manager.model_manager import ModelManager

class HomogenousFactory(AbstractFactory):
    '''    
    classdocs
    '''

    def __init__(self, datahandler, algorithm_factory):
        '''
        Constructor
        @param datahandler: datahandler object
        @param algorithm_class: class of algorithm, not instanciated
        @param alg_params: dict of parameters for given algorithm
        '''
        self.datahandler = datahandler
        self.alg_factory = algorithm_factory

    def get_instance(self):
        '''
        @return: model_manager
        '''
        feature_engineer = self.datahandler.get_feature_engineer()
        alg = self.alg_factory.get_instance()
        return ModelManager(alg, feature_engineer)

    #TODO: modify for mr
    def encode(self, manager):
        pass

    def decode(self, encoded):
        pass