'''
Created on Dec 4, 2013

@author: xapharius
'''

from algorithms import AbstractAlgorithmFactory
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
    
    def instanciate(self):
        '''Create a LinearRegression Object
        :return: Object implementing AbstractAlgorithm
        '''
        pass  