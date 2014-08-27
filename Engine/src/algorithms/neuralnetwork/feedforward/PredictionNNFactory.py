'''
Created on Feb 4, 2014

@author: xapharius
'''

from algorithms.AbstractAlgorithmFactory import AbstractAlgorithmFactory
from algorithms.neuralnetwork.feedforward.PredictionNN import PredictionNN, SimpleUpdate
import numpy as np
from algorithms.neuralnetwork.feedforward.BaggedPredictionNN import BaggedPredictionNN

class PredictionNNFactory(AbstractAlgorithmFactory):
    '''
    Factory class for Predictive Feed Forward Neural Network.
    Provides the functionalities specified by the AbstractAlgorithmClass.
    '''


    def __init__(self, arrLayerSizes, iterations=1, update_method=SimpleUpdate(0.5), batch_update_size=1):
        '''
        Initializes the Factory and sets the parameters for the Model
        '''
        self.arrLayerSizes = arrLayerSizes
        self.nrLayers = len(arrLayerSizes)
        self.iterations = iterations
        self.batch_update_size = batch_update_size
        self.update_method = update_method

    def get_instance(self):
        '''Create a PredictionNN Object
        :return: Object implementing AbstractAlgorithm
        '''
        return PredictionNN(self.arrLayerSizes, self.iterations, self.update_method, self.batch_update_size)

    def aggregate(self, NNArr):
        '''Aggregate all PredictionNN from NNArr Prameter by AVERAGING
        :param NNArr: (normal)array of PredictionNN
        :return combined PredictionNN
        '''
        
#         aggrWeightsArr = []
#         for layer in range(len(self.arrLayerSizes)-1):
#             weights = np.zeros((self.arrLayerSizes[layer]+1, self.arrLayerSizes[layer+1]))
#             aggrWeightsArr.append(weights)
#         
#         # for each network add respective layers
#         for NN in NNArr:
#             for wIndex in range(len(NN.weightsArr)):
#                 aggrWeightsArr[wIndex] += NN.weightsArr[wIndex]  
#         # divide by number of networks
#         for wIndex in range(len(NN.weightsArr)):
#                 aggrWeightsArr[wIndex] /= len(NNArr)
#         
#         aggrNN = self.get_instance()
#         aggrNN.set_params(aggrWeightsArr)
        bagged = BaggedPredictionNN(nns=NNArr)
        return bagged

    def encode(self, alg_instance):
        # bring nparrays to list type (to be json encodable)
        weightsArr = []
        try:
            for npArr in alg_instance.weightsArr:
                weightsArr.append(npArr.tolist())
        except AttributeError:
            for nn in alg_instance.nets:
                nn_weights = []
                for layer_weights in nn.weightsArr:
                    nn_weights.append(layer_weights.tolist())
                weightsArr.append(nn_weights)
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

