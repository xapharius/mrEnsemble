'''
Created on Feb 4, 2014

@author: xapharius
'''

from algorithms.AbstractAlgorithmFactory import AbstractAlgorithmFactory
from algorithms.neuralnetwork.feedforward.PredictionNN import PredictionNN
import numpy as np

class PredictionNNFactory(AbstractAlgorithmFactory):
    '''
    Factory class for Predictive Feed Forward Neural Network.
    Provides the functionalities specified by the AbstractAlgorithmClass.
    '''


    def __init__(self, arrLayerSizes):
        '''
        Initializes the Factory and sets the parameters for the Model
        '''
        self.arrLayerSizes = arrLayerSizes
    
    def get_instance(self):
        '''Create a PredictionNN Object
        :return: Object implementing AbstractAlgorithm
        '''
        newNN = PredictionNN(self.arrLayerSizes);
        return newNN
        
    def aggregate(self, NNArr):
        '''Aggregate all PredictionNN from NNArr Prameter by AVERAGING
        :param NNArr: (normal)array of PredictionNN
        :return combined PredictionNN
        '''
        
        aggrWeightsArr = []
        for layer in range(len(self.arrLayerSizes)-1):
            weights = np.zeros((self.arrLayerSizes[layer]+1, self.arrLayerSizes[layer+1]))
            aggrWeightsArr.append(weights)
        
        #for each network add respective layers
        for NN in NNArr:
            for wIndex in range(len(NN.weightsArr)):
                aggrWeightsArr[wIndex] += NN.weightsArr[wIndex]  
        #divide by numer of networks
        for wIndex in range(len(NN.weightsArr)):
                aggrWeightsArr[wIndex] /= len(NNArr)
        
        aggrNN = PredictionNN(self.arrLayerSizes)
        aggrNN.set_params(aggrWeightsArr)
        
        return aggrNN
    
    def encode(self, alg_instance):
        # bring nparrays to list type (to be json encodable)
        weightsArr = []
        for npArr in alg_instance.weightsArr:
            weightsArr.append(npArr.tolist())
        return weightsArr
    
    def decode(self, encoded):
        alg_list = []
        # encode is a list of encoded algorithm instances
        for alg_weightsArr in encoded:
            # turn each element of the list(list) into an npArray
            npWeightsArr = []
            for arr in alg_weightsArr:
                npWeightsArr.append(np.array(arr))
            # predictinNN can be created since we have a list of npArrays
            newPredictionNN = PredictionNN(self.arrLayerSizes)
            newPredictionNN.set_params(npWeightsArr)
            alg_list.append(newPredictionNN)
        return alg_list
            
        