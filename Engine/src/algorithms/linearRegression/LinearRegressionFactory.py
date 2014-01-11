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
        self.linRegArr = np.array([])
    
    def get_instance(self):
        '''Create a LinearRegression Object
        :return: Object implementing AbstractAlgorithm
        '''
        newLinReg = LinearRegression(self.nrLRparams);
        self.linRegArr = np.append(self.linRegArr, newLinReg)
        return newLinReg
        
    #TODO: Aggregation has to be done by Engine's reducer
    def aggregate(self):
        '''Aggregate all linRegs from linRegArr by AVERAGING
        :return combined linReg
        '''
        aggrLinRegParams = np.zeros([self.nrLRparams, ])
        for i in range(len(self.linRegArr)):
            aggrLinRegParams += self.linRegArr[i].params
        aggrLinRegParams /= len(self.linRegArr)
        
        aggrLinReg = LinearRegression(self.nrLRparams)
        aggrLinReg.set_params(aggrLinRegParams)
        
        return aggrLinReg
    
        
        