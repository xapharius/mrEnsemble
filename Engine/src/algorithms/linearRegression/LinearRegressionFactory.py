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


    def __init__(self, nrParams):
        '''
        Initializes the Factory and sets the parameters for the Model
        '''
        self.nrLRparams = nrParams
        self.nrModels = 0;
    
    def get_instance(self):
        '''Create a LinearRegression Object
        :return: Object implementing AbstractAlgorithm
        '''
        newLinReg = LinearRegression(self.nrLRparams);
        self.nrModels = self.nrModels + 1
        return newLinReg
        
    def aggregate(self, linRegArr):
        '''Aggregate all linRegs from linRegArr Prameter by AVERAGING
        :param linRegArr: np.array of LinearRegression
        :return combined linReg
        '''
        aggrLinRegParams = np.zeros([self.nrLRparams, ])
        
        for i in range(len(linRegArr)):
            aggrLinRegParams += linRegArr[i].params
        aggrLinRegParams /= len(linRegArr)
        
        aggrLinReg = LinearRegression(self.nrLRparams)
        aggrLinReg.set_params(aggrLinRegParams)
        
        return aggrLinReg
    
        
        