'''
Created on Dec 4, 2013

@author: xapharius
'''
from algorithms.AbstractAlgorithm import AbstractAlgorithm
import numpy as np 
import sys

class LinearRegression(AbstractAlgorithm):
    '''
    classdocs
    '''


    def __init__(self, _nrInputVars):
        '''
        Creates a linear Regression model - the parameters as a row vector
        :param nrInputs: number of latent variables Liner Model should have
        '''
        # add bias parameter
        self.params = np.random.rand(1,_nrInputVars+1)
        self.nrParams = self.params.size

    def train(self, _dataSet):
        '''
        Trains Model for given dataset
        Transactions for both inputs and targets should be as rows
        '''
        # add column of ones to dataset
        inputs = self.addOnes(_dataSet.inputs)
        pseudoInv = np.linalg.pinv(np.dot(inputs.T, inputs))
        part2 = np.dot(inputs.T, _dataSet.targets)
        self.params = np.dot(pseudoInv, part2).T
        print self.params.shape
        
    def addOnes(self, _matrix):
        '''
        @param param: np.array
        @return: same input matrix with an extra column of ones in front
        @rtype: np.array
        '''
        return np.append(np.ones([_matrix.shape[0], 1]), _matrix, 1)

    def predict(self, _dataSet):
        '''
        Predicts targets for given dataset.inputs
        @return: predictions
        @rtype: list of np.arrays
        '''
        inputs = self.addOnes(_dataSet.inputs)
        resMat = np.dot(inputs, self.params.T)
        retArr = []
        for i in range(resMat.shape[0]):
            retArr.append([np.array(resMat[i])])
        return retArr
            
    
    def set_params(self, parameters):
        #sys.stderr.write("set params: " + str(parameters) + "\n")
        '''Set parameters of predefined model(shape of parameters already specified)
        @param parameters: np.array
        @raise exception: if given parameters don't match in shape with model
        '''
        if self.params.shape != parameters.shape:
            raise Exception("overwriting parameters have not the same shape as the model.\n        model: " + self.params.shape + "\n  overwriting: " + str(parameters.shape))
        self.params = parameters
