'''
Created on Mar 23, 2015

@author: xapharius
'''
from factories.abstract_factory import AbstractFactory
from managers.model_manager import ModelManager

class HomogenousFactory(AbstractFactory):
    '''
    classdocs
    '''


    def __init__(self, datahandler, algorithm_class, alg_params = None):
        '''
        Constructor
        @param datahandler: datahandler object
        @param algorithm_class: class of algorithm, not instanciated
        @param alg_params: dict of parameters for given algorithm
        '''
        self.datahandler = datahandler
        self.alg_class = algorithm_class
        self.alg_params = alg_params

    def get_instance(self):
        '''
        @return: model_manager
        '''
        feature_selector = self.datahandler.get_feature_selector()
        alg = self.alg_class()
        return ModelManager(alg, feature_selector)

    #TODO: modify for mr
    def encode(self, manager):
        pass

    def decode(self, encoded):
        pass