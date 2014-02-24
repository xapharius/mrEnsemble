'''
Created on Dec 4, 2013

@author: xapharius
'''

from algorithms.AbstractAlgorithmFactory import AbstractAlgorithmFactory
from algorithms.linearRegression import LinearRegression
import numpy as np

class LinearRegressionFactory(AbstractAlgorithmFactory):
    '''
    Factory class for Linear Regression.
    Provides the functionalities specified by the AbstractAlgorithmClass.
    '''


    def __init__(self, nrInputVars):
        '''
        Initializes the Factory and sets the parameters for the Model
        '''
        self.nrInputVars = nrInputVars
        self.nrLRparams = nrInputVars+1
    
    def get_instance(self):
        '''Create a LinearRegression Object
        :return: Object implementing AbstractAlgorithm
        '''
        newLinReg = LinearRegression(self.nrInputVars);
        return newLinReg
        
    def aggregate(self, linRegArr):
        '''Aggregate all linRegs from linRegArr Prameter by AVERAGING
        :param linRegArr: (normal)array of LinearRegression
        :return combined linReg
        '''
        aggrLinRegParams = np.zeros([1, self.nrLRparams])
        
        for i in range(len(linRegArr)):
            aggrLinRegParams += linRegArr[i].params
        aggrLinRegParams /= len(linRegArr)
        
        aggrLinReg = LinearRegression(self.nrInputVars)
        aggrLinReg.set_params(aggrLinRegParams)
        
        return aggrLinReg
    
    def serialize(self, alg_instance):
        return alg_instance.params.tolist()
    
    def deserialize(self, serialized):
        deserialized = []
        for s in serialized:
            # observations are row vectors
            params = np.array(s).T
            lin_reg = LinearRegression(self.nrInputVars);
            lin_reg.set_params(params)
            deserialized.append(lin_reg)
        return deserialized
        