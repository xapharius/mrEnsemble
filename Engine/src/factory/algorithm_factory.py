'''
Created on Jul 28, 2015

@author: xapharius
'''
import random

class AlgorithmFactory(object):
    '''
    Generalized Model Factory
    '''

    def __init__(self, algorithm_class, algorithm_params=None):
        '''
        Constructor
        @param algoritm_class: class of the algorithm to be instantiated
        @param params: dictionary, param: list of values (only 1 value chosen per instance)
        '''
        self.algorithm_class = algorithm_class
        self.algorithm_params = algorithm_params if algorithm_params is not None else {}
        
    def get_random_params(self):
        dct = self.algorithm_params
        rdct = {}
        for key in dct.keys():
            values = dct[key]
            if type(values) == list:
                values = random.choice(values)
            rdct[key] = values
        return rdct

    def get_instance(self):
        '''
        sampling form parameters uniformly
        '''
        print self.get_random_params()
        return self.algorithm_class(**self.get_random_params())