'''
Created on Dec 4, 2013

@author: xapharius
'''
from algorithms.AbstractAlgorithm import AbstractAlgorithm
import numpy as np 
from numpy.f2py.auxfuncs import throw_error

class LinearRegression(AbstractAlgorithm):
    '''
    classdocs
    '''


    def __init__(self, _nrParams):
        '''
        Creates a linear Regression model - the parameters as a row vector
        :param nrParams: number of parameters Liner Model should have
        '''
        self.params = np.random.rand(_nrParams,)
        self.nrParams = self.params.size

    #TODO: may raise exception when trying to invert singular matrix        
    def train(self, _dataSet):
        '''
        Trains Model for given dataset
        Transactions for both inputs and targets should be as rows
        '''
        self.params = (_dataSet.inputs.T * _dataSet.inputs).I * _dataSet.inputs.T * _dataSet.targets
    

    def predict(self, _dataSet):
        '''
        Predicts targets for given dataset.inputs
        '''
        return self.params * _dataSet.inputs
    
    def set_params(self, parameters):
        '''Set parameters of predefined model(shape of parameters already specified)
        @param parameters: np.array
        @raise exception: if given parameters don't match in shape with model
        '''
        if self.nrParams != parameters.size:
            raise Exception("overwriting parameters have not same shape as model")
        self.params = parameters
        