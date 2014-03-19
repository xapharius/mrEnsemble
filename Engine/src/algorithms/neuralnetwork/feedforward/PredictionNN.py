'''
Created on Feb 4, 2014

@author: xapharius
'''
from algorithms.AbstractAlgorithm import AbstractAlgorithm
import numpy as np 
import utils.numpyutils as nputils

class PredictionNN(AbstractAlgorithm):
    '''
    Predictive Feed Forward Neural Network Class
    '''


    def __init__(self, arrLayerSizes):
        '''
        Creates a Prediction Neural Network - weights 
        :param arrLayerSizes: list with number of neurons each layer should have. index starts with input layer.
        '''
        # Sizes for each layer, 0 is input layer
        self.arrLayerSizes = arrLayerSizes
        self.nrLayers = len(arrLayerSizes)
        
        weightsArr = []
        for layer in range(len(arrLayerSizes)-1):
            # weight matrix shape is first layer * second layer
            # bias term added on first dimension
            # generate random weights in the range of [-0.5, 0.5]
            weights = np.random.rand(arrLayerSizes[layer]+1, arrLayerSizes[layer+1])- 0.5
            weightsArr.append(weights)
        
        self.weightsArr = weightsArr
    
    def train(self, dataSet):
        '''
        Online Training for given dataset
        @param _dataSet: NumericalDataSet
        '''
        # go through all observation
        for inputArr, targetArr in dataSet.gen_observations():
            # feedforward
            activations = self.feedforward(inputArr)
            # backprop
            self.backpropagation(activations, targetArr)

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
            # bias values are not included in the activations
            activations.append(currActivations)
            # add bias to outputs
            currActivations = nputils.addOneToVec(currActivations)
            
        return activations
        
    def sigDer(self, x):
        return x * (1 - x)
    
    def backpropagation(self, activations, targets, learningRate = 0.5):
        '''
        Propagates errors through NN, computing the partial gradients and updating the weights
        @param activations: List of np.arrays obtained from feedforward
        @param targets: np.array of shape (1,output_layer_size) representing the desired output
        @param learningRate: scalar. speed at which we move down the gradient  
        '''
        # the deltas from the delta rule
        # deltas for the output layer - no sigmoid derivative since output is linear
        deltaRule = [(targets - activations[-1]).transpose()] 
        # starting from second last layer, iterating backwards through NN
        # weights i are between layer i and i + 1
        for i in reversed(range(1, self.nrLayers-1)):
            # multiply weights with previous computed deltas (first in list) to obtain 
            # the sum over all neurons in next layer, for each neuron in current layer
            sums = np.dot(self.weightsArr[i], deltaRule[0])
            # remove last sum since it is from bias neuron. we don't need a delta for it, 
            # since it doesn't have connections to the previous layer
            sums = sums[:-1,:]
            # element-wise multiply with the sigmoidal derivative for activation
            deltaRule_thisLayer = self.sigDer(activations[i]).transpose() * sums
            #PREpend deltas to array  
            deltaRule = [deltaRule_thisLayer] + deltaRule
            
        # update weights
        # weights i are between layer i and i + 1
        for i in range(self.nrLayers-1):
            # here the activations need the additional bias neuron -> addOneToVec
            # unfortunately both arrays have to be transposed
            weightsChange = learningRate * np.dot(deltaRule[i], nputils.addOneToVec(activations[i])).transpose()
            self.weightsArr[i] = self.weightsArr[i] + weightsChange

