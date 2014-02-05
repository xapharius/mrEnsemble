'''
Created on Feb 4, 2014

@author: xapharius
'''
from algorithms.AbstractAlgorithm import AbstractAlgorithm
import numpy as np 
import sys

class PredictionNN(AbstractAlgorithm):
    '''
    Predictive Feed Forward Neural Network Class
    '''


    def __init__(self, arrLayerSizes):
        '''
        Creates a Prediction Neural Network - weights 
        :param arrLayerSizes: number of neurons each layer should have. index starts with input layer.
        '''
        #Sizes for each layer, 0 is input layer
        self.arrLayerSizes = arrLayerSizes
        
        weightsArr = []
        for layer in range(len(arrLayerSizes)-1):
            weights = np.random.rand(arrLayerSizes[layer], arrLayerSizes[layer+1])
            weightsArr = [weightsArr, weights]
        
        self.weightsArr = weightsArr
    
    # TODO: train        
    def train(self, _dataSet):
        '''
        Trains Model for given dataset
        Transactions for both inputs and targets should be as rows
        '''
        #feedforward
        #backprop
        pass

    # TODO: predict
    def predict(self, _dataSet):
        '''
        Predicts targets for given dataset.inputs
        '''
        #feedforward
        pass
    
    def set_params(self, weightsArr):
        '''Set parameters of predefined model(shape of parameters already specified)
        @param parameters: array of np.array
        @raise exception: if given parameters don't match in shape with model
        '''
        for wIndex in range(len(weightsArr)):
            if self.weightsArr[wIndex].shape != weightsArr[wIndex].shape:
                raise Exception("overwriting parameters have not same shape as model on weight Matrix " + str(wIndex) + ".\n        model: " + str(self.weightsArr[wIndex].shape) + "\n  overwriting: " + str(weightsArr[wIndex].shape))
            self.weightsArr[wIndex] = weightsArr[wIndex]
        
    # TODO: feedforward
    # TODO: backprop

