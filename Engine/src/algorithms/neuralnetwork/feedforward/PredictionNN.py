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
        # Sizes for each layer, 0 is input layer
        self.arrLayerSizes = arrLayerSizes
        
        weightsArr = []
        for layer in range(len(arrLayerSizes)-1):
            # weight matrix shape is first layer * second layer
            # bias term added on first dimension
            weights = np.random.rand(arrLayerSizes[layer]+1, arrLayerSizes[layer+1])
            weightsArr.append(weights)
        
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
    
    def set_params(self, parameters):
        '''Set parameters of predefined model(shape of parameters already specified)
        @param parameters: array of np.array
        @raise exception: if given parameters don't match in shape with model
        '''
        for wIndex in range(len(parameters)):
            if self.weightsArr[wIndex].shape != parameters[wIndex].shape:
                raise Exception("overwriting parameters have not the same shape as the model (weight Matrix) " + str(wIndex) + ".\n        model: " + str(self.weightsArr[wIndex].shape) + "\n  overwriting: " + str(parameters[wIndex].shape))
            self.weightsArr[wIndex] = parameters[wIndex]
        
    # TODO: feedforward
    # TODO: backprop

