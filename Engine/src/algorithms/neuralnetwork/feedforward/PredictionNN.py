'''
Created on Feb 4, 2014

@author: xapharius
'''
from algorithms.AbstractAlgorithm import AbstractAlgorithm
import numpy as np 
import utils.numpyutils as nputils
import sys
import datahandler.numerical.NumericalDataSet

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

    def predict(self, dataSet):
        '''
        Predicts targets for given dataset
        @param _dataSet: data Set inheriting AbstractDataSet
        @return: outputs from the feed forward on each row 
        @rtype: list of numpy.ndarray (nr_obs * nr_output_neurons)
        '''
        predictions = [];
        # loop through dataset
        for observation, _ in dataSet.gen_observations():
            # make sure it numpy array
            inputArr = np.array(observation) 
            # feedforward
            activations = self.feedforward(inputArr)
            # extract output
            output = activations[len(activations)-1]
            
            predictions.append(output)
        
        return predictions
        
    def set_params(self, parameters):
        '''Set parameters of predefined model(shape of parameters already specified)
        @param parameters: array of np.array
        @raise exception: if given parameters don't match in shape with model
        '''
        for wIndex in range(len(parameters)):
            if self.weightsArr[wIndex].shape != parameters[wIndex].shape:
                raise Exception("overwriting parameters have not the same shape as the model (weight Matrix) " + str(wIndex) + ".\n        model: " + str(self.weightsArr[wIndex].shape) + "\n  overwriting: " + str(parameters[wIndex].shape))
            self.weightsArr[wIndex] = parameters[wIndex]
        
    def feedforward(self, inputVec):
        '''
        feed inputs forward through net.
        @param: inputVec nparray of inputs. Size defined by input layer. Row vector shape = (1,x) hint: np.array([[1,2,3]])
        @return: activations for each neuron.
        @rtype: array of np.Arrays(1dim), for each layer one (weight layers + 1)
        @raise exception: if given input size doesn't match with input layer
        '''
        
        if (inputVec.shape != (1, self.arrLayerSizes[0])):
            raise Exception("Invalid inputvector shape. (1,"+str(self.arrLayerSizes[0])+") expected, got " + str(inputVec.shape))
        
        activations = [];
        
        activations.append(inputVec)
        currActivations = nputils.addOneToVec(inputVec)
        
        # feed forward through network
        for i in range(len(self.weightsArr)):
            # weighted sum for each neuron
            currActivations = np.dot(currActivations, self.weightsArr[i])
            # activation function is a logistic unit, except last layer
            if i != len(self.weightsArr)-1:
                currActivations = nputils.sigmoidNPArray(currActivations)
            # TODO: bias activations also in activations array?
            activations.append(currActivations)
            # add bias to outputs
            currActivations = nputils.addOneToVec(currActivations)
            
        return activations
        
    
    # TODO: backprop

