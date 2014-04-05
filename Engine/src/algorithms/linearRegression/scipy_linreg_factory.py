'''
Created on Apr 5, 2014

@author: Simon
'''
from algorithms.AbstractAlgorithmFactory import AbstractAlgorithmFactory
from scipy_linreg import SciPyLinReg
import numpy as np

class SciPyLinRegFactory(AbstractAlgorithmFactory):
    '''
    classdocs
    '''


    def __init__(self, _type=SciPyLinReg.ORDINARY, alpha=0.3):
        '''
        Constructor
        '''
        self.type = _type
        self.alpha = alpha

    def aggregate(self, models_arr):
        aggregated = np.zeros(models_arr[0].get_params().shape)
        for model in models_arr:
            aggregated += model.get_params()
        aggregated /= len(models_arr)
        
        result = SciPyLinReg(self.type, self.alpha)
        result.set_params(aggregated)
        return result

    def encode(self, alg_instance):
        return alg_instance.get_params().tolist()

    def decode(self, encoded):
        decoded_list = []
        for params in encoded:
            decoded = SciPyLinReg(self.type, self.alpha)
            decoded.set_params(params)
            decoded_list.append(decoded)
        return decoded_list

    def get_instance(self):
        return SciPyLinReg(self.type, self.alpha)
