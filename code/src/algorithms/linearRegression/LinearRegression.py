'''
Created on Dec 4, 2013

@author: xapharius
'''
from algorithms.AbstractAlgorithm import AbstractAlgorithm
import numpy as np 

class LinearRegression(AbstractAlgorithm):
    '''
    classdocs
    '''


    def __init__(self, _nrParams):
        '''
        Creates a linear Regression model - the parameters as a row vector
        :param nrParams: number of parameters Liner Model should have
        '''
        self.params = np.random.rand(1, _nrParams)

    #TODO may raise exception at trying to invert singular matrix        
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
        