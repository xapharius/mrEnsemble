'''
Created on Feb 4, 2014

@author: xapharius
'''

from algorithms.AbstractAlgorithmFactory import AbstractAlgorithmFactory
from algorithms.neuralnetwork.feedforward.multilayer_perceptron import MultilayerPerceptron, SimpleUpdate
import numpy as np
from algorithms.neuralnetwork.feedforward.BaggedPredictionNN import BaggedPredictionNN
import utils.numpyutils as nputils

class PredictionNNFactory(AbstractAlgorithmFactory):
    '''
    Factory class for Predictive Feed Forward Neural Network.
    Provides the functionalities specified by the AbstractAlgorithmClass.
    '''


    def __init__(self, arrLayerSizes, iterations=1, update_method=SimpleUpdate(0.5), batch_update_size=1, activ_func=(nputils.sigmoid_np_arr, nputils.sigmoid_deriv_np_arr), do_classification=False):
        '''
        Initializes the Factory and sets the parameters for the Model
        '''
        self.arrLayerSizes = arrLayerSizes
        self.nrLayers = len(arrLayerSizes)
        self.iterations = iterations
        self.batch_update_size = batch_update_size
        self.update_method = update_method
        self.activ_func = activ_func
        self.do_classification = do_classification

    def get_instance(self):
        '''Create a PredictionNN Object
        :return: Object implementing AbstractAlgorithm
        '''
        return MultilayerPerceptron(self.arrLayerSizes, iterations=self.iterations, update_method=self.update_method, batch_update_size=self.batch_update_size, activ_func=self.activ_func, do_classification=self.do_classification)

    def aggregate(self, NNArr):
        '''Aggregate all PredictionNN from NNArr Parameter by AVERAGING
        :param NNArr: (normal)array of PredictionNN
        :return combined PredictionNN
        '''
        
        aggrWeightsArr = []
        for layer in range(len(self.arrLayerSizes)-1):
            weights = np.zeros((self.arrLayerSizes[layer]+1, self.arrLayerSizes[layer+1]))
            aggrWeightsArr.append(weights)
         
        # for each network add respective layers
        for NN in NNArr:
            for wIndex in range(len(NN.weights_arr)):
                aggrWeightsArr[wIndex] += NN.weights_arr[wIndex]  
        # divide by number of networks
        for wIndex in range(len(NN.weights_arr)):
                aggrWeightsArr[wIndex] /= len(NNArr)
         
        aggrNN = self.get_instance()
        aggrNN.set_params(aggrWeightsArr)
        return aggrNN

    def encode(self, alg_instance):
        # bring nparrays to list type (to be json encodable)
        weightsArr = []
        # serialization of a single multilayer perceptron
        for npArr in alg_instance.weights_arr:
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
            newPredictionNN = MultilayerPerceptron(self.arrLayerSizes)
            newPredictionNN.set_params(npWeightsArr)
            alg_list.append(newPredictionNN)
        return alg_list

